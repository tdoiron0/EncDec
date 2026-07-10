#!/bin/bash

usage() {
    echo "Usage: $0 --server <user@host> --port <ssh-port>" >&2
    echo "e.g.   $0 --server root@216.243.220.222 --port 11686" >&2
    exit 1
}

SERVER=""
SERVER_PORT=""

while [ $# -gt 0 ]; do
    case "$1" in
        --server) SERVER="${2:?--server requires a value}"; shift 2 ;;
        --port|-p) SERVER_PORT="${2:?--port requires a value}"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

if [ -z "$SERVER" ] || [ -z "$SERVER_PORT" ]; then
    usage
fi

SERVER_SSH_PEM_PATH="~/.ssh/id_ed25519"
SERVER_INSTALL_PATH="/workspace"

TEMP_KEY_DESC="temp-clone-key"
TEMP_KEY_FILENAME="temp-key"
REPO_OWNER="tdoiron0"
REPO="EncDec"
DEPLOY_KEY_ID=""

cleanup() {
    status=$?
    set +e
    
    echo "Cleaning up"
    rm "$TEMP_KEY_FILENAME" "$TEMP_KEY_FILENAME.pub" "src/datasets/data-processed.zip" "tokenizers/tokenizers.zip"

    # The deploy key and the server-side copy of the keypair are kept so the
    # server retains read access to the repo. Revoke when destroying the server:
    if [ -n "$DEPLOY_KEY_ID" ]; then
        echo "Deploy key left registered. Revoke with:"
        echo "  gh api --method DELETE repos/$REPO_OWNER/$REPO/keys/$DEPLOY_KEY_ID"
    fi

    # Debug cleanup because many tests
    # gh api --method DELETE "repos/$REPO_OWNER/$REPO/keys/$DEPLOY_KEY_ID"
    # ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "rm ~/.ssh/$TEMP_KEY_FILENAME"
    # ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "rm -rf $SERVER_INSTALL_PATH/$REPO"
}

trap cleanup EXIT INT TERM

echo "Generating new temporary read only SSH keypair"
ssh-keygen -q -t ed25519 -N "" -C "$TEMP_KEY_DESC" -f "$TEMP_KEY_FILENAME"

echo "Registering temporary SSH keypair"
DEPLOY_KEY_ID="$(
    gh api --method POST "repos/$REPO_OWNER/$REPO/keys" \
        -f title="$TEMP_KEY_DESC" \
        -f key="$(<"$TEMP_KEY_FILENAME.pub")" \
        -F read_only=true \
        --jq '.id'
)"
echo "DEPLOY KEY ID=$DEPLOY_KEY_ID"

echo "Uploading temporary SSH keypair to server"
scp -i "$SERVER_SSH_PEM_PATH" -P "$SERVER_PORT" "$TEMP_KEY_FILENAME" "$SERVER:~/.ssh/$TEMP_KEY_FILENAME"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "chmod 700 ~/.ssh"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "chmod 600 ~/.ssh/$TEMP_KEY_FILENAME"

echo "Cloneing repo on external server"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "git clone git@github.com:$REPO_OWNER/$REPO.git $SERVER_INSTALL_PATH/$REPO --config core.sshCommand=\"ssh -i ~/.ssh/$TEMP_KEY_FILENAME -o StrictHostKeyChecking=accept-new\""

echo "Zipping local datasets"
cd "src/datasets"
zip -r data-processed.zip "data-processed"
cd "../.."

echo "Zipping local tokenizers"
cd "tokenizers"
zip tokenizers.zip *.model *.vocab
cd ".."

echo "Uploading local datasets"
scp -i "$SERVER_SSH_PEM_PATH" -P "$SERVER_PORT" "src/datasets/data-processed.zip" "$SERVER:$SERVER_INSTALL_PATH/$REPO/src/datasets"

echo "Uploading local tokenizers"
scp -i "$SERVER_SSH_PEM_PATH" -P "$SERVER_PORT" "tokenizers/tokenizers.zip" "$SERVER:$SERVER_INSTALL_PATH/$REPO"

echo "Installing system packages on server"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" 'bash -s' <<'REMOTE'
set -e
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq zip unzip curl git
REMOTE

echo "Unzipping server datasets"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "unzip $SERVER_INSTALL_PATH/$REPO/src/datasets/data-processed.zip -d $SERVER_INSTALL_PATH/$REPO/src/datasets"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "rm -rf $SERVER_INSTALL_PATH/$REPO/src/datasets/data-processed.zip"

echo "Unzipping server tokenizers"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "unzip $SERVER_INSTALL_PATH/$REPO/Models/Jap2Eng/tokenizers.zip -d $SERVER_INSTALL_PATH/$REPO/tokenizers"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "rm -rf $SERVER_INSTALL_PATH/$REPO/tokenizers.zip"

echo "Installing Miniconda on server"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" 'bash -s' <<'REMOTE'
set -e

MINICONDA_PATH="$HOME/miniconda3"

if [ -d "$MINICONDA_PATH" ]; then
    echo "Miniconda already installed at $MINICONDA_PATH"
else
    # curl/wget may be missing on a fresh image
    command -v curl >/dev/null 2>&1 || { apt-get update && apt-get install -y curl; }

    INSTALLER="/tmp/miniconda.sh"
    curl -fsSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o "$INSTALLER"

    # -b = batch/non-interactive, -p = install prefix
    bash "$INSTALLER" -b -p "$MINICONDA_PATH"
    rm -f "$INSTALLER"

    # make `conda` available in future non-login shells
    "$MINICONDA_PATH/bin/conda" init bash
fi
REMOTE

echo "Installing dependencies"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" 'bash -s' "$SERVER_INSTALL_PATH" "$REPO" <<'REMOTE'
set -e
SERVER_INSTALL_PATH="$1"
REPO="$2"
source "/root/miniconda3/etc/profile.d/conda.sh"

conda activate base

pip install -r "$SERVER_INSTALL_PATH/$REPO/requirements.txt"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

REMOTE

ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "cd $SERVER_INSTALL_PATH/$REPO"
ssh -i "$SERVER_SSH_PEM_PATH" -p "$SERVER_PORT" "$SERVER" "pip install -e ."