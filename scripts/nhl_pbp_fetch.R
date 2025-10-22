# Fetch NHL play-by-play (PBP) and shifts using SportsDataverse packages
# Saves Parquet files into data/raw/nhl_pbp/

# Usage (PowerShell):
#   Rscript scripts/nhl_pbp_fetch.R --seasons 2019 2020 2021 2022 2023 2024 2025 --out data/raw/nhl_pbp

# Use project-local library to avoid admin installs
proj_lib <- file.path(getwd(), 'data', 'r_libs')
if (!dir.exists(proj_lib)) dir.create(proj_lib, recursive = TRUE)
.libPaths(unique(c(proj_lib, .libPaths())))
options(pkgType = 'binary', repos = 'https://cloud.r-project.org')

ensure_pkg <- function(name) {
  if (!requireNamespace(name, quietly = TRUE)) {
    install.packages(name, lib = proj_lib)
  }
}

ensure_pkg('optparse')
ensure_pkg('arrow')

suppressPackageStartupMessages({
  library(optparse)
  library(arrow)
})

# Try both names; some environments use fastRhockey for NHL
.resolve_pbp_loader <- function() {
  # Try fastRhockey variants
  if (requireNamespace("fastRhockey", quietly = TRUE)) {
    fx <- getNamespaceExports("fastRhockey")
    if ("load_nhl_pbp" %in% fx) return(list(pkg = "fastRhockey", fun = fastRhockey::load_nhl_pbp))
    if ("load_pbp" %in% fx) return(list(pkg = "fastRhockey", fun = fastRhockey::load_pbp))
  }
  # Try hockeyR
  if (requireNamespace("hockeyR", quietly = TRUE)) {
    hx <- getNamespaceExports("hockeyR")
    if ("nhl_pbp" %in% hx) return(list(pkg = "hockeyR", fun = hockeyR::nhl_pbp))
  }
  # Try installing fastRhockey then retry
  ensure_pkg('fastRhockey')
  if (requireNamespace("fastRhockey", quietly = TRUE)) {
    fx <- getNamespaceExports("fastRhockey")
    if ("load_nhl_pbp" %in% fx) return(list(pkg = "fastRhockey", fun = fastRhockey::load_nhl_pbp))
    if ("load_pbp" %in% fx) return(list(pkg = "fastRhockey", fun = fastRhockey::load_pbp))
  }
  stop("No suitable PBP loader found in fastRhockey or hockeyR.")
}

.resolve_shifts_loader <- function() {
  if (requireNamespace("fastRhockey", quietly = TRUE)) {
    fx <- getNamespaceExports("fastRhockey")
    if ("load_shifts" %in% fx) return(list(pkg = "fastRhockey", fun = fastRhockey::load_shifts))
  }
  return(NULL)
}

pbp_loader <- .resolve_pbp_loader()
shifts_loader <- .resolve_shifts_loader()

opt_list <- list(
  make_option(c("--seasons"), type = "character", action = "store", default = "2023 2024 2025",
              help = "Space-separated list of seasons (e.g., 2019 2020 2021)"),
  make_option(c("--out"), type = "character", default = "data/raw/nhl_pbp",
              help = "Output directory for Parquet files")
)

opt <- parse_args(OptionParser(option_list = opt_list))
seasons <- as.integer(strsplit(opt$seasons, "[ ,]+")[[1]])
out_dir <- opt$out
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

message("[pbp] Using package: ", pbp_loader$pkg)
message("[pbp] Seasons: ", paste(seasons, collapse = ", "))
message("[pbp] Output dir: ", out_dir)

load_pbp <- pbp_loader$fun
load_shifts <- if (!is.null(shifts_loader)) shifts_loader$fun else NULL

for (s in seasons) {
  message("[pbp] Fetching PBP for season ", s)
  # Support either 'seasons' or 'season' parameter names
  arg_names <- names(formals(load_pbp))
  args <- list()
  if (!is.null(arg_names)) {
    if ("seasons" %in% arg_names) args$seasons <- s else if ("season" %in% arg_names) args$season <- s else args[[1]] <- s
  } else {
    args[[1]] <- s
  }
  pbp <- tryCatch(do.call(load_pbp, args), error = function(e) {
    message("[warn] Failed load_pbp for ", s, ": ", e$message); NULL
  })
  if (!is.null(pbp)) {
    p_out <- file.path(out_dir, sprintf("pbp_%d.parquet", s))
    arrow::write_parquet(pbp, p_out)
    message("[pbp] Wrote ", p_out)
  }
  if (!is.null(load_shifts)) {
    message("[shifts] Fetching shifts for season ", s)
    arg_names_s <- names(formals(load_shifts))
    args_s <- list()
    if (!is.null(arg_names_s)) {
      if ("seasons" %in% arg_names_s) args_s$seasons <- s else if ("season" %in% arg_names_s) args_s$season <- s else args_s[[1]] <- s
    } else {
      args_s[[1]] <- s
    }
    shifts <- tryCatch(do.call(load_shifts, args_s), error = function(e) {
      message("[warn] Failed load_shifts for ", s, ": ", e$message); NULL
    })
    if (!is.null(shifts)) {
      s_out <- file.path(out_dir, sprintf("shifts_%d.parquet", s))
      arrow::write_parquet(shifts, s_out)
      message("[shifts] Wrote ", s_out)
    }
  }
}

message("[done] NHL PBP/shifts export complete")
