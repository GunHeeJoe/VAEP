python DL_Binary_model.py \
--data binary \
--model entitiy_dcn \
--epoch 50 \
--learning_rate 1e-4 \
--batch_size 4096 \
--weight_decay 1e-2 \
--devicd cuda \
--save_dir chkpt 

python DL_Multiclass_model.py \
--data multiclass \
--model entitiy_dcn \
--epoch 50 \
--learning_rate 1e-4 \
--batch_size 4096 \
--weight_decay 1e-2 \
--devicd cuda \
--save_dir chkpt 



