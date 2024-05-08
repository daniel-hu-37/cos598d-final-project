export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

python3 run_glue_dist.py \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --local_rank 0\
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir

  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --overwrite_output_dir