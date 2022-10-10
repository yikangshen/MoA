OMP_NUM_THREADS=4
OUTPUT_DIR=checkpoints/experiment_$( date +%Y-%m-%d_%H-%M-%S )

# Train the model
mkdir -p ${OUTPUT_DIR}
fairseq-train $1 \
    --arch moa_transformer_base --share-all-embeddings --amp \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-9 --clip-norm 0.0 \
    --warmup-init-lr 1e-07 --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.2 --attention-dropout 0.2 --activation-dropout 0.1 --weight-decay 0 \
    --num-expert 8 --encoder-attention-heads 8 --decoder-attention-heads 8 --head-dim 128 \
    --sample-topk 0 --cvloss 0 --switchloss 0.01 --zloss 0.001 --gating normal \
    --criterion moa_ce_loss --label-smoothing 0.1 \
    --tensorboard-logdir ${OUTPUT_DIR} \
    --update-freq 32 \
    --user-dir ./ \
    --max-tokens 8192 --max-epoch 100 \
    --save-dir ${OUTPUT_DIR} \
    --keep-best-checkpoints 5 \
    --keep-last-epochs 10 \
    --log-interval 10 \
    --ffd-type normal