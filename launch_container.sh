#!/bin/bash
set -e
set -o pipefail

function warning()
{
    yellow="33m"
    echo -e "\033[${yellow}[$(date +'%Y-%m-%dT%H:%M:%S%z')] WARNING: $* \033[0m" 2>&1
}

function error() {
    local red="31m"
    echo -e "\033[${red}[$(date +'%Y-%m-%dT%H:%M:%S%z')] ERROR: $* \033[0m" 1>&2
    exit 1
}

function info() {
    green="32m"
    echo -e "\033[${green}[$(date +'%Y-%m-%dT%H:%M:%S%z')] INFO: $* \033[0m" 2>&1
}

function usage() {
    echo "Usage: bash launch_container.sh"
    echo 
    echo "      -h, --help        show helps"
    exit 1
}

LCCL_HOME="/u/hyang/lccl"
MPI_HOME="/u/hyang/openmpi"

function launch_container() {
    if which git > /dev/null 2>&1; then
        NCCL_TESTS_HOME=$( git rev-parse --show-toplevel )
    else
        NCCL_TESTS_HOME="${PWD}"
        if basename "${NCCL_TESTS_HOME}" != "nccl-tests"; then
            error "export NCCL_TESTS_HOME=<path-to-nccl-tests>"
        fi
    fi

    while (($#)); 
    do
        case "$1" in
        *)
          usage
          ;;
        esac
    done

    # check docker
    if ! which docker &>/dev/null; then
        error "command 'docker' not found"
    fi

    local image=nccl-tests:ubuntu2204
    cp -f "${NCCL_TESTS_HOME}/requirements.txt" "${NCCL_TESTS_HOME}/docker/ubuntu2204/requirements.txt"
    docker build -t "${image}" -f "${NCCL_TESTS_HOME}/docker/ubuntu2204/Dockerfile" "${NCCL_TESTS_HOME}/docker/ubuntu2204"
    docker rm -f nccl-tests || true
    docker create --name nccl-tests \
        --gpus all \
        -it \
        --rm \
        --privileged \
        --network host \
        "${image}" \
        bash
    docker cp "${NCCL_TESTS_HOME}" nccl-tests:/
    if [[ -d "${LCCL_HOME}" ]]; then
        #build lccl
        docker cp "${LCCL_HOME}/build" nccl-tests:/lccl
    fi
    # if [[ -d "${MPI_HOME}" ]]; then
    #     docker cp "${MPI_HOME}/build" nccl-tests:/mpi
    # fi
    docker start -i nccl-tests
}

launch_container "$@"