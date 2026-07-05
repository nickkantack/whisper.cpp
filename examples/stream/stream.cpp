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

// command-line parameters
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
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;
    bool flash_attn    = true;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-bs"   || arg == "--beam-size")     { params.beam_size     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"   || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-nfa"  || arg == "--no-flash-attn") { params.flash_attn    = false; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help          [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n",    params.n_threads);
    fprintf(stderr, "            --step N        [%-7d] audio step size in milliseconds\n",                params.step_ms);
    fprintf(stderr, "            --length N      [%-7d] audio length in milliseconds\n",                   params.length_ms);
    fprintf(stderr, "            --keep N        [%-7d] audio to keep from previous step in ms\n",         params.keep_ms);
    fprintf(stderr, "  -c ID,    --capture ID    [%-7d] capture device ID\n",                              params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N  [%-7d] maximum number of tokens per audio chunk\n",       params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N   [%-7d] audio context size (0 - all)\n",                   params.audio_ctx);
    fprintf(stderr, "  -bs N,    --beam-size N   [%-7d] beam size for beam search\n",                      params.beam_size);
    fprintf(stderr, "  -vth N,   --vad-thold N   [%-7.2f] voice activity detection threshold\n",           params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N  [%-7.2f] high-pass frequency cutoff\n",                   params.freq_thold);
    fprintf(stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n",      params.translate ? "true" : "false");
    fprintf(stderr, "  -nf,      --no-fallback   [%-7s] do not use temperature fallback while decoding\n", params.no_fallback ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special [%-7s] print special tokens\n",                           params.print_special ? "true" : "false");
    fprintf(stderr, "  -kc,      --keep-context  [%-7s] keep context between audio chunks\n",              params.no_context ? "false" : "true");
    fprintf(stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n",                                params.language.c_str());
    fprintf(stderr, "  -m FNAME, --model FNAME   [%-7s] model path\n",                                     params.model.c_str());
    fprintf(stderr, "  -f FNAME, --file FNAME    [%-7s] text output file name\n",                          params.fname_out.c_str());
    fprintf(stderr, "  -tdrz,    --tinydiarize   [%-7s] enable tinydiarize (requires a tdrz model)\n",     params.tinydiarize ? "true" : "false");
    fprintf(stderr, "  -sa,      --save-audio    [%-7s] save the recorded audio to a file\n",              params.save_audio ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu        [%-7s] disable GPU inference\n",                          params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn    [%-7s] enable flash attention during inference\n",        params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,     --no-flash-attn [%-7s] disable flash attention during inference\n",       params.flash_attn ? "false" : "true");
    fprintf(stderr, "\n");
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
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio

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

    // Start streaming audio to the buffer
    audio.resume();

    // whisper init
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

    // n_iter merely counts the number of times the whisper model
    // was invoked. It only appears in print statements.
    int n_iter = 0;
    bool is_running = true;

    printf("[Start speaking]\n");
    fflush(stdout);

    /*
    t_last is the latest time when vad_simple() returned true.
    I believe it is the last time a falling edge of audio activity
    was detected, i.e. silence after speaking.
    */
    auto t_last  = std::chrono::high_resolution_clock::now();

    /*
    t_start is the time the transcription started. It is never
    updated during the program's lifetime.
    */
    const auto t_start = t_last;

    // main audio loop
    while (is_running) {

        // handle Ctrl + C
        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        // process new audio
        const auto t_now  = std::chrono::high_resolution_clock::now();
        const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

        /*
        Stop processing if it has been fewer than 2 seconds since
        the last falling edge of speech activity (i.e. the last time
        the whisper model was called). This logic also has the effect
        of clamping the rate of whisper inference invocations at a 
        max of once every two seconds.
        */
        if (t_diff < 2000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        /*
        We use the last 2 seconds of audio to try to detect the falling
        edge of speech activity. Note that audio.get will resize pcmf32_new
        to the size needed for the requested duration of audio.
        */
        audio.get(2000, pcmf32_new);
        if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
            // TODO update
            /*
            For our application, passing the last length_ms audio anytime speech
            stopped is too clunky, and will result in the same speech being
            transcribed multiple times if the utterances are short and length_ms
            is long. It would be better to request ((t_diff) / 1000000) milliseconds
            (i.e. audio.get(t_diff / 1000000, pcmf32)), and even better if we 
            also detected the last rising edge of speech activity (say, t_silence_broken)
            and then used (t_now - t_silence_broken) / 1000000. The only catch to
            this approach is that if someone is talking continuously, this duration
            could cause us to ask for more audio than the buffer contains, since
            we declare at the start how much audio to buffer. It also runs the risk
            of asking the whisper model to run inference on a huge swath of audio.

            To solve, this we can detect when we have maxed the audio buffer and
            run inference on that, knowing that we easily could have clipped the
            speaker's current word. This gives us a transcript of the last x seconds
            (x being the buffer length) with perhaps some errors around speech
            within the last second. Knowing that we cut the speaker off, we can
            plan our next audio range to pass to the whipser model to contain
            some overlap with the audio we grabbed now. We'd take a task downstream
            to try to stitch together the transcript.

            I think the basic logic is:
            is_speaking = false;
            rising_edge_time = 0;
            falling_edge_time = 0;
            while (is_running) {
                if (!is_speaking && detect_rising_edge()) {
                    rising_edge_time = now - 2000 (or however big the detection window is)
                    is_speaking = true
                    continue;
                }
                if (is_speaking && detect_falling_edge()) {
                    falling_edge_time = now
                    is_speaking = false
                    // Run inference on audio from rising_edge_time to falling_edge_time
                }
                if (is_speaking && now - rising_edge_time close to buffer length) {
                    // Run inference on audio from rising_edge_time to now
                    rising_edge_time = now - 2000 (some sensible overlap)
                }
            }
            */
            audio.get(params.length_ms, pcmf32);
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // t_last stores the latest time for which we decided the 2 seconds
        // leading up to t_last contained a falling edge of speech activity.
        t_last = t_now;

        // run the inference
        {
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

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                return 6;
            }

            // print result;
            {
                /*
                t1 is a DURATION, not a time. The whisper model was just
                invoked, and t1 is the time since the program's start
                until the moment right before the whisper model's most
                recent inference run.
                t1 should roughly equal "time since program start" as
                long as the whisper model doesn't take too long.
                */
                const int64_t t1 = (t_last - t_start).count()/1000000;
                /*
                t0 is length_ms before t1. It is determined by the reasoning
                "I just detected an utterance of length x ms, so it must have
                started x ms ago".
                */
                const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                printf("\n");
                printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                printf("\n");

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    /*
                    I think t0 and t1 below are timestamps relative
                    to the window defined by t0 and t1 above.

                    TODO try printing to_timestamp(t0_above + t0, false)
                    instead of to_timestamp(t0, false)
                    */
                    const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                    const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                    std::string output = "[" + to_timestamp(t0, false) + " --> " + to_timestamp(t1, false) + "]  " + text;

                    // I think for my use-case this is always false
                    if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                        output += " [SPEAKER_TURN]";
                    }

                    output += "\n";

                    printf("%s", output.c_str());
                    fflush(stdout);

                }

                printf("\n");
                printf("### Transcription %d END\n", n_iter);
            }

            ++n_iter;

            fflush(stdout);
        }
    }

    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return 0;
}
