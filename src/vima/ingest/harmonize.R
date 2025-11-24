#!/usr/bin/env Rscript

library(harmony)
library(arrow)
library(parallel)

set.seed(0)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
    stop("Usage: Rscript harmonize.R <ncores> <input.feather> <metadata1> [metadata2 ...]")
}

requested_ncores <- as.integer(args[1])
infile <- args[2]
metadata_names <- args[3:length(args)]

# --- Detect available cores (handles SLURM if present) ---
slurm_cores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", unset = NA))
system_cores <- detectCores()
available_cores <- if (!is.na(slurm_cores)) slurm_cores else system_cores
if (is.na(available_cores)) {
    available_cores <- 1
}
ncores <- min(requested_ncores, available_cores)
message("Detected available cores: ", available_cores)
if (requested_ncores > ncores) {
    message("Requested ncores = ", requested_ncores,
            " but only ", available_cores, " available. Using ncores = ", ncores, ".")
}

stem <- tools::file_path_sans_ext(infile)
ext <- tools::file_ext(infile)
outfile <- paste0(stem, "_harmony.", ext)

cat("Input file:", infile, "\n")
cat("Metadata:", paste(metadata_names, collapse = ", "), "\n")

data <- read_feather(infile)
pcs <- data[, grep("^PC", names(data))]
metadata <- data[, metadata_names]

cat("Running Harmony...\n")

harmPcs <- try(
    harmony::RunHarmony(
        pcs,
        metadata,
        metadata_names,
        verbose = TRUE,
        ncores = ncores
    ),
    silent = TRUE
)

if (inherits(harmPcs, "try-error")) {
    message("\nHarmony failed unexpectedly.")
    q(status = 1)
}

cat("Harmony completed successfully.\n")
cat("Writing output to:", outfile, "\n")

write_feather(cbind(as.data.frame(harmPcs), metadata), outfile)

cat("Done.\n")