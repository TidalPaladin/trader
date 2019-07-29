#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
#set -o xtrace
# shellcheck disable=SC1091

#tensorboard --logdir=/artifacts/tblogs &
exec "$@"
