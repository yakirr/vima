library(harmony)
library(arrow)

args <- commandArgs(trailingOnly=TRUE)
infile = args[1]
metadata_names = args[2:length(args)]
stem <- tools::file_path_sans_ext(infile)
ext <- tools::file_ext(infile)
outfile <- paste0(stem, "_harmony", ".", ext)

cat('harmonizing\n')
cat('metadata names:', metadata_names, '\n')

data <- read_feather(infile)
pcs <- data[, setdiff(names(data), metadata_names)]
metadata = data[, metadata_names]

harmPcs <- harmony::RunHarmony(
    pcs, metadata, metadata_names, verbose=TRUE, ncores=8
)

cat('writing ', outfile, '\n')
write_feather(cbind(as.data.frame(harmPcs), metadata), outfile)