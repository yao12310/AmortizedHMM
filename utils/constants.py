# Dataframe Cols
METHYL_BED_COLS = [
    'contig', 'start_pos', 'end_pos',
    'name', 'score', 'strand',
    'start_codon', 'end_codon', 'rgb',
    'coverage', 'methyl_pct'
] # for na12878 methyl ref, see https://www.encodeproject.org/data-standards/wgbs/
FASTA_FAI_COLS = [
    'contig', 'num_bases', 'byte_index',
    'bases_per_line', 'bytes_per_line'
] # for fasta.fai, see https://www.biostars.org/p/98885/
NANOPOLISH_BINARY_SUMMARY_COLS = [
    'contig', 'strand', 'pos', 'kmer', 'read_name', 'log_lik_ratio', 'singleton', # call-methylation output
    'coverage', 'methyl_pct', 'label' # methylation label data
] # for summarizing nanopolish methylation calling results
DEEPSIGNAL_EXTRACT_COLS = [
    'contig', 'pos', 'strand', 'pos_in_strand', 'readname', 'read_strand', 'kmer',
    'signal_means', 'signal_std', 'signal_len', 'cent_signal', 'label'
] # for deepsignal feature extraction
DEEPSIGNAL_CALL_COLS = [
    'contig', 'pos', 'strand', 'pos_in_strand', 'readname', 'read_strand',
    'prob_0', 'prob_1', 'called_label', 'k_mer'
] # for deepsignal call_mods output
DEEPSIGNAL_CALL_LABEL_COLS = [
    'contig', 'pos', 'strand', 'pos_in_strand', 'readname', 'read_strand',
    'prob_0', 'prob_1', 'called_label', 'k_mer', 'coverage', 'methyl_pct'
] # for deepsignal call_mods + labels output

# Misc
KMER_LEN = 6
