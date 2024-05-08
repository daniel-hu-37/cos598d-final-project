export GLUE_DIR=$HOME/glue_data
export TASK_NAME=RTE

python3 task2b/run_glue.py \
  --model_type xlnet \
  --model_name_or_path xlnet-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --local_rank 1\
  --world_size 4\
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/ \
  --master_ip 10.10.1.2 \
  --master_port 5555 \
  --overwrite_output_dir