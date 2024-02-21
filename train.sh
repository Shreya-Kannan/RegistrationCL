#!/bin/bash
#SBATCH --account=rrg-punithak
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --mem=32G
#SBATCH --time=40:00:00
#SBATCH --mail-user=skannan3@ualberta.ca
#SBATCH --mail-type=ALL

module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -e /home/shreya/scratch/voxelmorph
pip install torch torchvision
pip install numpy==1.23.0

NOW=$(date '+%Y%m%d%H%M%S')
#python /home/shreya/scratch/voxelmorph/scripts/torch/train.py --img-list '/home/shreya/scratch/voxelmorph/images/AbdomenMRCT/AbdomenMRCT_dataset.json'  --epochs 3 --image-loss "ncc" --lambda 1 --model-dir /home/shreya/scratch/voxelmorph/models/$NOW --load-model /home/shreya/scratch/voxelmorph/models/20240131155443/0500.pt
python /home/shreya/scratch/voxelmorph/scripts/torch/train.py --img-list '/home/shreya/scratch/voxelmorph/images/AbdomenMRCT/AbdomenMRCT_dataset.json'  --epochs "$1" --image-loss "$2" --cl-type "$3" --lambda 1 --random-mode True --filt-size "$4" --n-features "$5" --model-dir /home/shreya/scratch/voxelmorph/models/"$2"_"$3"_"$1"_"$4"_"$5"_$NOW 
#python /home/shreya/scratch/voxelmorph/scripts/torch/test.py --json '/home/shreya/scratch/voxelmorph/images/AbdomenMRCT/AbdomenMRCT_dataset.json' --set val --model /home/shreya/scratch/voxelmorph/models/cl_f-micl_1000_20240206165738/1000.ptcd