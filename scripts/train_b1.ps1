$ErrorActionPreference = "Stop"

$ConfigPath = if ($args.Count -ge 1) { $args[0] } else { "configs/config.yaml" }

python train.py --config $ConfigPath --baseline b1
