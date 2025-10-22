# Install required R packages for NHL PBP pipeline into a user-writable library
required <- c('fastRhockey','nhlfastR','arrow','optparse')
repos <- 'https://cloud.r-project.org'

# Use project-local library to avoid admin permissions
proj_lib <- file.path(getwd(), 'data', 'r_libs')
if (!dir.exists(proj_lib)) dir.create(proj_lib, recursive = TRUE)
.libPaths(unique(c(proj_lib, .libPaths())))

message('[r] Using library path: ', proj_lib)
for (pkg in required) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = repos, lib = proj_lib)
  }
}
cat('[r] package installation complete\n')
