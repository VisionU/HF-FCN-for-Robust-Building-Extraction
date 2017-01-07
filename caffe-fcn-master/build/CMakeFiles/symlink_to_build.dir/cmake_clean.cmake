FILE(REMOVE_RECURSE
  "CMakeFiles/symlink_to_build"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/symlink_to_build.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
