# Global variables
. setup/config.sh

# Delete folder
rm -rf ${PROJECT}

# Create folder
mkdir ${PROJECT}

# Copy executable
cp bin/miner ${PROJECT}/
cp bin/miner_nvidia ${PROJECT}/
cp bin/miner_amd ${PROJECT}/

# Copy kernels
cp -r bin/kernel ${PROJECT}/

# Zip folder
tar czvf ${PROJECT}-${VERSION}.tar.gz ${PROJECT}
