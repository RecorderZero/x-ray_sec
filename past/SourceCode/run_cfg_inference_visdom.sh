#!/usr/bin/env bash
set -euo pipefail

# Run one CFG-DDIM inference job and stream its input/output/difference images
# to Visdom. Start ./start_visdom_cfg_ddim.sh in another terminal first.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

usage() {
    cat <<'EOF'
Usage:
  ./run_cfg_inference_visdom.sh [OPTIONS]

Options:
  --data-dir DIR        Image directory (default: data/CheXpert-v1.0/valid)
  --model-path FILE     CFG checkpoint
  --result-dir DIR      Output prefix (the sampler appends YYYY_MM_DD)
  --num-samples N       Number of images (default: 1)
  --noise-level N       DDIM inversion noise level (default: 500)
  --guidance-scale X    Classifier-free guidance scale (default: 4.0)
  --timesteps VALUE     Timestep respacing (default: ddim1000)
  --visdom-env NAME     Visdom environment; default is a timestamped name
  --check               Validate environment and paths without inference
  -h, --help            Show this help

Environment overrides:
  CONDA_ENV, VISDOM_SERVER, VISDOM_PORT, VISDOM_ENV, DATA_DIR, MODEL_PATH,
  RESULT_DIR, NUM_SAMPLES, NOISE_LEVEL, GUIDANCE_SCALE, TIMESTEP_RESPACING.

Example (two terminals):
  ./start_visdom_cfg_ddim.sh
  ./run_cfg_inference_visdom.sh --data-dir data/my_xrays --num-samples 1
EOF
}

conda_env="${CONDA_ENV:-CFG_DDIM}"
visdom_server="${VISDOM_SERVER:-http://127.0.0.1}"
visdom_port="${VISDOM_PORT:-8850}"
visdom_env="${VISDOM_ENV:-cfg_ddim_$(date +%Y%m%d_%H%M%S)}"
data_dir="${DATA_DIR:-data/CheXpert-v1.0/valid}"
model_path="${MODEL_PATH:-results/Model/cfg_chexpert_p_uncond_0.1_v1_2025_05_08/modelchexpert050000.pt}"
result_dir="${RESULT_DIR:-results/visdom_cfg_inference}"
num_samples="${NUM_SAMPLES:-1}"
noise_level="${NOISE_LEVEL:-500}"
guidance_scale="${GUIDANCE_SCALE:-4.0}"
timestep_respacing="${TIMESTEP_RESPACING:-ddim1000}"
check_only=0

while (($#)); do
    case "$1" in
        --data-dir) data_dir="${2:?--data-dir requires a directory}"; shift 2 ;;
        --model-path) model_path="${2:?--model-path requires a file}"; shift 2 ;;
        --result-dir) result_dir="${2:?--result-dir requires a directory}"; shift 2 ;;
        --num-samples) num_samples="${2:?--num-samples requires a value}"; shift 2 ;;
        --noise-level) noise_level="${2:?--noise-level requires a value}"; shift 2 ;;
        --guidance-scale) guidance_scale="${2:?--guidance-scale requires a value}"; shift 2 ;;
        --timesteps) timestep_respacing="${2:?--timesteps requires a value}"; shift 2 ;;
        --visdom-env) visdom_env="${2:?--visdom-env requires a name}"; shift 2 ;;
        --check) check_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$data_dir" != /* ]]; then
    data_dir="$script_dir/$data_dir"
fi
if [[ "$model_path" != /* ]]; then
    model_path="$script_dir/$model_path"
fi
if [[ "$result_dir" != /* ]]; then
    result_dir="$script_dir/$result_dir"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in PATH." >&2
    exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required for the Visdom health check." >&2
    exit 1
fi
if [[ ! -f "$model_path" ]]; then
    echo "ERROR: checkpoint not found: $model_path" >&2
    exit 1
fi
if [[ ! -d "$data_dir" ]]; then
    echo "ERROR: image directory not found: $data_dir" >&2
    exit 1
fi

first_image="$(find "$data_dir" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.npy' \) -print -quit)"
if [[ -z "$first_image" ]]; then
    echo "ERROR: no supported images found under: $data_dir" >&2
    echo "       Pass the real image folder with --data-dir DIR." >&2
    exit 1
fi
if [[ ! "$num_samples" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --num-samples must be a positive integer." >&2
    exit 1
fi
if [[ ! "$noise_level" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --noise-level must be a non-negative integer." >&2
    exit 1
fi

visdom_url="${visdom_server%/}:$visdom_port"
if ! curl --silent --show-error --fail --max-time 3 "$visdom_url" >/dev/null; then
    echo "ERROR: Visdom is not reachable at $visdom_url" >&2
    echo "       Start it in another terminal:" >&2
    echo "       $script_dir/start_visdom_cfg_ddim.sh" >&2
    exit 1
fi

echo "CFG-DDIM inference configuration"
echo "  data:        $data_dir"
echo "  checkpoint:  $model_path"
echo "  samples:     $num_samples"
echo "  noise:       $noise_level"
echo "  guidance:    $guidance_scale"
echo "  timesteps:   $timestep_respacing"
echo "  output:      ${result_dir}_$(date +%Y_%m_%d)"
echo "  Visdom:      $visdom_url (environment: $visdom_env)"

if ((check_only)); then
    echo "Check passed; inference was not started."
    exit 0
fi

export VISDOM_SERVER="$visdom_server"
export VISDOM_PORT="$visdom_port"
export VISDOM_ENV="$visdom_env"

model_flags=(
    --unet_version v1
    --image_size 256
    --in_channels 1
    --num_channels 128
    --num_classes 2
    --class_cond True
    --num_res_blocks 2
    --num_heads 1
    --learn_sigma True
    --use_scale_shift_norm False
    --attention_resolutions 16
)

diffusion_flags=(
    --diffusion_steps 1000
    --noise_schedule linear
    --rescale_learned_sigmas False
    --rescale_timesteps False
)

sample_flags=(
    --batch_size 1
    --num_samples "$num_samples"
    --timestep_respacing "$timestep_respacing"
    --use_ddim True
)

conda run --no-capture-output -n "$conda_env" \
    python scripts/cfg_image_sample.py \
    --data_dir "$data_dir" \
    --result_dir "$result_dir" \
    --model_path "$model_path" \
    --dataset chexpert \
    --noise_level "$noise_level" \
    --guidance_scale "$guidance_scale" \
    "${model_flags[@]}" \
    "${diffusion_flags[@]}" \
    "${sample_flags[@]}"

echo
echo "Inference complete."
echo "Open $visdom_url and select environment: $visdom_env"
echo "Saved files: ${result_dir}_$(date +%Y_%m_%d)"
