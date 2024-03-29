# Global variables
. setup/config.sh

# Delete folder
rm -rf ${PROJECT}

# Create folder
mkdir ${PROJECT}

# Copy executable
cp bin/miner ${PROJECT}/

# Copy kernels
cp -r bin/Release/kernel ${PROJECT}/

# Zip folder
tar czvf ${PROJECT}-${VERSION}.tar.gz ${PROJECT}
