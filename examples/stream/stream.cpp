// Real-time speech recognition of input from a microphone
//
// A very quick-n-dirty implementation serving mainly as a proof of concept.
//
#include "common-sdl.h"
#include "common.h"
#include "common-whisper.h"
#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <iomanip>

struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t beam_size  = -1;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = true;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            printf("We will not help you.");
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(0);
        }
    }

    return true;
}

static std::string get_system_timestamp() {
    auto system_time_as_time_t = 
        std::chrono::system_clock::to_time_t(
            std::chrono::system_clock::now()
        );
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&system_time_as_time_t), "%F %T");
    std::string timestamp_string = oss.str();
    return timestamp_string;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "\n");
}

std::string make_uuid() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());

    uint64_t a = gen();
    uint64_t b = gen();

    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(8) << (uint32_t)(a >> 32) << '-'
        << std::setw(4) << (uint16_t)(a >> 16) << '-'
        << std::setw(4) << (uint16_t)a << '-'
        << std::setw(4) << (uint16_t)(b >> 48) << '-'
        << std::setw(12) << (b & 0x0000FFFFFFFFFFFFULL);

    return oss.str();
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // Initialize some values that inform the whisper model of our
    // use case.
    const bool use_vad = true;
    params.no_timestamps  = !use_vad;
    params.max_tokens     = 0;

    /*
    Define an audio buffer with a length matching length_ms. This
    means audio.get allows us to request up to the last length_ms
    of audio.
    */
    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    params.language = "en";

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    std::vector<float> pcmf32;
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new;

    // TODO could initialize with domain specific vocab
    std::vector<whisper_token> prompt_tokens;
    bool use_prompt = true;
    std::string prompt_text = "This is a transcript of radio chatter. It often includes references to \"SAR\", acronym for Search and Rescue.";
    {
        prompt_tokens.resize(1024);
        const int n = whisper_tokenize(ctx, prompt_text.c_str(), prompt_tokens.data(), 1024);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to tokenize prompt '%s'\n", __func__, prompt_text.c_str());
            return 4;
        }
        prompt_tokens.resize(n);
    }

    bool is_running = true;

    std::string event = "{\"type\": \"InitializationEvent\",\"time\":\"" + get_system_timestamp() + "\"}\n";
    printf("%s", event.c_str());
    fflush(stdout);

    auto last_inference_time = std::chrono::high_resolution_clock::now();

    const auto program_start_time = last_inference_time;

    auto speech_start_time_cursor = 
        std::chrono::high_resolution_clock::now();
    auto speech_start_approximate_system_time_cursor = 
        std::chrono::system_clock::now();
    auto next_speech_start_time_cursor = 
        std::chrono::high_resolution_clock::now();
    auto next_speech_start_approximate_system_time_cursor = 
        std::chrono::system_clock::now();

    bool is_speaking = false;

    wav_writer wavWriter;

    // main audio loop
    while (is_running) {

        // handle Ctrl + C
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        const int decision_interval_ms = 2000;
        const int min_ms_between_loops = 100;

        // process new audio
        const auto t_now  = std::chrono::high_resolution_clock::now();
        const auto time_since_last_inference = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - last_inference_time).count();
        const auto system_clock_now = std::chrono::system_clock::now();

        /*
        Decisions about whether to transcribe are made every 2 seconds.
        continue if not time to make a decision yet.
        */
        if (time_since_last_inference < decision_interval_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(min_ms_between_loops));
            continue;
        }

        // last_inference_time stores the latest time for which we made a transcription
        // decision.
        last_inference_time = t_now;

        /*
        We use the last 2 seconds of audio to try to detect the falling
        edge of speech activity. Note that audio.get will resize pcmf32_new
        to the size needed for the requested duration of audio.
        */
        audio.get(decision_interval_ms, pcmf32_new);

        bool do_run_inference = false;
        bool was_speaker_cut_off = false;
        const VadState vad_state = get_vad_state(
            pcmf32_new, WHISPER_SAMPLE_RATE, decision_interval_ms / 2, params.vad_thold, 
            params.freq_thold, false);

        if (!is_speaking && vad_state == VadState::ActivityStart) {
            speech_start_time_cursor = t_now - std::chrono::milliseconds(decision_interval_ms);
            speech_start_approximate_system_time_cursor = system_clock_now -
                std::chrono::milliseconds(decision_interval_ms);
            is_speaking = true;
            event = "{\"type\": \"SpeechStartEvent\",\"time\":\"" + get_system_timestamp() + "\"}\n";
            printf("%s", event.c_str());
            fflush(stdout);
            continue;
        }
        if (is_speaking && vad_state == VadState::ActivityEnd) {
            next_speech_start_time_cursor = t_now;
            next_speech_start_approximate_system_time_cursor = system_clock_now;
            is_speaking = false;
            do_run_inference = true;
            event = "{\"type\": \"SpeechStopEvent\",\"time\":\"" + get_system_timestamp() + "\"}\n";
            printf("%s", event.c_str());
            fflush(stdout);
        } else if (is_speaking && (t_now - speech_start_time_cursor).count() / 1E6 > params.length_ms - decision_interval_ms) {
            next_speech_start_time_cursor = t_now - std::chrono::milliseconds(decision_interval_ms);
            next_speech_start_approximate_system_time_cursor = system_clock_now - 
                std::chrono::milliseconds(decision_interval_ms);
            was_speaker_cut_off = true;
            do_run_inference = true;
        }

        if (!do_run_inference) {
            continue;
        }

        audio.get((int) ((t_now - speech_start_time_cursor).count() / 1E6), pcmf32);

        whisper_full_params wparams = whisper_full_default_params(params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

        wparams.print_progress   = false;
        wparams.print_special    = params.print_special;
        wparams.print_realtime   = false;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.translate        = params.translate;
        wparams.single_segment   = !use_vad;
        wparams.max_tokens       = params.max_tokens;
        wparams.language         = params.language.c_str();
        wparams.n_threads        = params.n_threads;
        wparams.beam_search.beam_size = params.beam_size;
        wparams.audio_ctx        = params.audio_ctx;
        wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]
        wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
        wparams.prompt_tokens    = use_prompt ? nullptr : prompt_tokens.data();
        wparams.prompt_n_tokens  = use_prompt ? 0       : prompt_tokens.size();

        std::string event = "{\"type\": \"InterferenceStartEvent\",\"time\":\"" + get_system_timestamp() + "\"}\n";
        printf("%s", event.c_str());
        fflush(stdout);
        if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
            fprintf(stderr, "%s: failed to process audio\n", argv[0]);
            return 6;
        }
        event = "{\"type\": \"InterferenceStopEvent\",\"time\":\"" + get_system_timestamp() + "\"}\n";
        printf("%s", event.c_str());
        fflush(stdout);

        auto speech_system_start_to_time_t = 
            std::chrono::system_clock::to_time_t(speech_start_approximate_system_time_cursor);

        {
            const int n_segments = whisper_full_n_segments(ctx);
            for (int i = 0; i < n_segments; ++i) {
                const char * text = whisper_full_get_segment_text(ctx, i);

                /* TODO - https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp#L376
                Shows a method to computer t0 and t1 which are stop and start times relative to the 
                speech_system_start_to_time_t. Use these to save only the audio (the portion of pcmf32)
                that participated in the segment.
                */

                // t0 and t1 are in units of hundredths of seconds
                const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                const int64_t start_index = t0 * WHISPER_SAMPLE_RATE / 100;
                const int64_t end_index = t1 * WHISPER_SAMPLE_RATE / 100;

                const std::vector<float> relevant_pcmf32(pcmf32.begin() + start_index, pcmf32.begin() + end_index);

                std::ostringstream oss;
                oss << std::put_time(std::localtime(&speech_system_start_to_time_t), "%F %T");
                std::string timestamp_string = oss.str();

                std::string output = "{\"type\":\"SegmentEvent\", \"time\": \"" + 
                    timestamp_string + "\", \"text\":\"" + text + (was_speaker_cut_off ? " (continuing...)" : "") + "\"}\n";

                if (strcmp(text, " [BLANK_AUDIO]") != 0) {

                    {
                        std::string uuid = make_uuid();
                        char filename[64];
                        snprintf(filename, sizeof(filename), "segment_%s.wav", uuid.c_str());
                        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
                        wavWriter.write(relevant_pcmf32.data(), relevant_pcmf32.size());
                        wavWriter.close();
                        std::ofstream out("segment_" + uuid + ".txt");
                        out << output;
                    }

                    printf("%s", output.c_str());
                    fflush(stdout);
                }
            }
        }

        fflush(stdout);

        speech_start_time_cursor = next_speech_start_time_cursor;
        speech_start_approximate_system_time_cursor = 
            next_speech_start_approximate_system_time_cursor;
    }

    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
