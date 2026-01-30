# Global variables
. setup/config.sh


# Delete folder
rm -rf ${PROJECT}

# Create folder
mkdir ${PROJECT}

# Copy config hiveos
cp setup/hiveos/h-config.sh ${PROJECT}/
cp setup/hiveos/h-run.sh ${PROJECT}/
cp setup/hiveos/h-stats.sh ${PROJECT}/
cp setup/hiveos/h-manifest.conf ${PROJECT}/

# Add right executable
chmod +x ${PROJECT}/h-config.sh
chmod +x ${PROJECT}/h-run.sh
chmod +x ${PROJECT}/h-stats.sh

# Copy executable
cp bin/miner ${PROJECT}/

# Copy kernels
cp -r bin/kernel ${PROJECT}/

# Zip folder
tar czvf ${PROJECT}-${VERSION}_hiveos.tar.gz ${PROJECT}
