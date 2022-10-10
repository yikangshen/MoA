CHECKPOINT_PATH=$1
DATA=wmt14_en_de_joined_dict
SPLIT=test
TRANSFORMER_CLINIC_ROOT=/path/to/Transformer-Clinic
FAIRSEQ_ROOT=/path/to/fairseq
mosesdecoder=$TRANSFORMER_CLINIC_ROOT/pre-process/mosesdecoder
tok_gold_targets=$TRANSFORMER_CLINIC_ROOT/pre-process/wmt14_en_de/tmp/$SPLIT.de
decodes_file=$CHECKPOINT_PATH/generate/generate-${SPLIT}.cleaned

# Average checkpoints
python3 $FAIRSEQ_ROOT/scripts/average_checkpoints.py --inputs $CHECKPOINT_PATH/ --num-epoch-checkpoints 10 --output $CHECKPOINT_PATH/averaged_model.pt

# Generate
fairseq-generate \
    $TRANSFORMER_CLINIC_ROOT/data-bin/${DATA}\
    --path $CHECKPOINT_PATH/averaged_model.pt \
    --remove-bpe --lenpen 0.6 --beam 4 \
    --gen-subset $SPLIT \
    --user-dir . --results-path $CHECKPOINT_PATH/generate

#Clean outputs
python clean_generate.py --generate_file $CHECKPOINT_PATH/generate/generate-$SPLIT.txt --output_file $CHECKPOINT_PATH/generate/generate-$SPLIT.cleaned

# Replace unicode.
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $decodes_file > $decodes_file.n
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $tok_gold_targets > $decodes_file.gold.n

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.gold.n > $decodes_file.gold.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.n > $decodes_file.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $decodes_file.gold.atat < $decodes_file.atat

# # Detokenize.
# perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l de < $decodes_file.n > $decodes_file.detok
# perl $mosesdecoder/scripts/tokenizer/detokenizer.perl -l de < $decodes_file.gold.n > $decodes_file.gold.detok

# sacrebleu $decodes_file.gold.detok -l en-de -i $decodes_file.detok -b
