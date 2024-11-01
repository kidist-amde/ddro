#!/bin/sh
#SBATCH --job-name=Eval_pretrain_url
##SBATCH --partition gpu_h100
##SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64gb #120gb #180gb
#SBATCH -c 16
#SBATCH --output=logs-slurm/eval-logs/Eval_pretrain_url-%j.out # %j is the job ID

# Set up the environment.
# source /home/kmekonnen/.bashrc
source ${HOME}/.bashrc
conda activate dsi-env
cd /gpfs/work4/0/prjs1037/dpo-exp/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/test_nq_t5_url.py \
                --encoding url
#!/bin/sh
#SBATCH --job-name=Eval_pretrain_search_url
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
##SBATCH --gres=gpu:nvidia_l40:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00 # d-h:m:s
#SBATCH --mem=64gb # memory per GPU 
#SBATCH -c 16 # number of CPUs
#SBATCH --output=logs-slurm/eval-logs/url/Eval_pretrain_search_url-%j.out # %j is the job ID
# Set up the environment.

source /home/kmekonn/.bashrc
conda activate ddro
cd /ivi/ilps/personal/kmekonn/projects/DDRO-Direct-Document-Relevance-Optimization/ddro
nvidia-smi 

python utils/test_nq_t5_url.py \
                --encoding url