worker_nodes=$1
nproc_per_node=$2
mode=$3
loop_num=$4
timestamp=$(date "+%Y-%m-%d-%H-%M-%S")

set -o xtrace
sco acp jobs create \
	--workspace-name=lsm3dv \
	--aec2-name=lsm3dv \
	--job-name=sco_inference_hunyuan_720p_${mode} \
	--priority=HIGH \
    --container-image-url='registry.cn-fz-01.fjscms.com/iag04fj02-ccr/vigen:v1' \
    --storage-mount='0ff4e551-7e44-11ef-8f95-7ac3b2306a9f:/mnt/afs' \
    --training-framework=pytorch \
    --worker-nodes=${worker_nodes} \
    --worker-spec="N4lS.Iq.I80.${nproc_per_node}" \
    --command="source activate /mnt/afs/miniconda-cuda_12_4/envs/sparsevideogen; \
    cd /mnt/afs/chenguojun/Sparse-VideoGen; \
    bash scripts/hyvideo_inference.sh $worker_nodes $nproc_per_node $mode $loop_num 2>&1 | tee logs/${timestamp}-inference.log"

