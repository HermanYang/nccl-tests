#!/bin/bash
set -e
set -o pipefail

function info() {
    green="32m"
    echo -e "\033[${green}[$(date +'%Y-%m-%dT%H:%M:%S%z')] INFO: $* \033[0m" 2>&1
}

function parse_args() {
    MODE="Multi-Thread"
    DRY_RUN=False
    DEVICE_IDS="0"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --mode)
                MODE=$2
                shift 2
                ;;
            --device_ids)
                DEVICE_IDS=$2
                shift 2
                while [[ $# -gt 0 && ! $1 =~ "--" ]]; do
                    DEVICE_IDS="$DEVICE_IDS $1"
                    shift 1
                done
                ;;
            --dry_run)
                DRY_RUN=True
                shift 1
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --mode: [Multi-Process] or [Multi-Thread]"
                echo "  --device_ids: 0 1 2 3 4 5 6 7 ..."
                echo "  --dry_run: Dry run"
                exit 0
                ;;
            *)
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    done
}

function config() {
    info "Mode: $MODE"
    if $(ls /sys/devices | grep -q "iommu"); then
        export HAS_IOMMU=yes
    else
        export HAS_IOMMU=no
    fi
    info "HAS IOMMU: $HAS_IOMMU"
    col_op_list="sendrecv gather scatter broadcast reduce reduce_scatter all_gather all_reduce alltoall hypercube"
    info "Collective Operators: $col_op_list"
    device_cpu_node_id_list=""
    for device_id in $DEVICE_IDS;
    do
        device_cpu_node_id_list="$device_cpu_node_id_list $(nvidia-smi topo -i $device_id -C | awk '{print $NF}')"
    done
    info "Device IDs: $DEVICE_IDS"
    info "Device CPU Node IDs: $device_cpu_node_id_list"
    info "Card Total Number: $(echo $DEVICE_IDS | wc -w)"
}

function build() {
    info "Building..."
    NCCL_HOME=""
    MPI_HOME=""
    if [[ -d "/lccl" ]]; then
        NCCL_HOME="/lccl"
    fi
    if [[ -d "/mpi" ]]; then
        MPI_HOME="/mpi"
    fi

    if [ -z ${MPI_HOME} ]; then
        if [ -z ${NCCL_HOME} ]; then
            make MPI=0
        else
            make MPI=0 NCCL_HOME="${NCCL_HOME}"
            export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH}"
        fi
    else
        if [ -z ${NCCL_HOME} ]; then
            make MPI=1 MPI_HOME="${MPI_HOME}"
            export LD_LIBRARY_PATH="${MPI_HOME}/lib:${LD_LIBRARY_PATH}"
        else
            make MPI=1 MPI_HOME="${MPI_HOME}" NCCL_HOME="${NCCL_HOME}"
            export LD_LIBRARY_PATH="${MPI_HOME}/lib:${NCCL_HOME}/lib:${LD_LIBRARY_PATH}"
        fi
    fi
}

function install_python_packages() {
    info "Installing Python packages..."
    python3 -m pip install -r requirements.txt
}

function test() {
    info "Testing..."
    numanode_involvement_list=$(echo $device_cpu_node_id_list | tr ' ' '\n' | sort | uniq -c | awk '{print $2}')
    pid_list=""
    for col_op in $col_op_list; do
        info "$col_op"
        for node_id in $numanode_involvement_list; do
            device_ids=""
            for device_id in $DEVICE_IDS; do
                if [ "$(nvidia-smi topo -i $device_id -C | awk '{print $NF}')" == "$node_id" ]; then
                    device_ids="$device_ids $device_id"
                fi
            done
            info "Device ID $device_ids Running on Node $node_id"
            if [ "$DRY_RUN" == "True" ]; then
                echo "numactl --cpunodebind=$node_id --membind=$node_id python3 run.py --nccl_tests_dir . --op ${col_op} --device_ids ${device_ids} --output_dir output/$col_op/node_${node_id}"
            else
                numactl --cpunodebind=$node_id --membind=$node_id python3 run.py --nccl_tests_dir . --op ${col_op} --device_ids ${device_ids} --output_dir output/$col_op/node_${node_id} &
                pid_list="$pid_list $!"
            fi
        done
        for pid in $pid_list; do
            wait $pid
        done
    done
}

parse_args $@
build
install_python_packages
config
test