set(OUT_CRYPTO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/kernel/crypto)

# Créer le répertoire de sortie s'il n'existe pas
file(MAKE_DIRECTORY ${OUT_CRYPTO})

# Liste des fichiers à copier
set(CRYPTO_FILES
  blake2b.cl
  blake2b_compress.cl
  fnv1.cl
  keccak_f1600.cl
  keccak_f800.cl
  kiss99.cl
)

# Boucle pour générer les commandes personnalisées
foreach(FILE ${CRYPTO_FILES})
  add_custom_command(
    OUTPUT ${OUT_CRYPTO}/${FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} ${OUT_CRYPTO}/${FILE}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${FILE}
    COMMENT "Copy ${FILE}"
  )
endforeach()

# Créer une cible qui dépend de tous les fichiers copiés
add_custom_target(copy_crypto_opencl_files ALL
  DEPENDS
    ${OUT_CRYPTO}/blake2b.cl
    ${OUT_CRYPTO}/blake2b_compress.cl
    ${OUT_CRYPTO}/fnv1.cl
    ${OUT_CRYPTO}/keccak_f1600.cl
    ${OUT_CRYPTO}/keccak_f800.cl
    ${OUT_CRYPTO}/kiss99.cl
)