./toolkits/collaborative_filtering/gensgd --training=rec_log_train.txt --val_pos=2 --rehash=1 --limit_rating=1000000 --max_iter=10 --gensgd_mult_dec=0.999999 --minval=-1 --maxval=1 --quiet=1  --calc_error=1 --file_columns=4
./toolkits/collaborative_filtering/gensgd --training=rec_log_train.txt --val_pos=2 --rehash=1 --limit_rating=1000000 --max_iter=100 --gensgd_mult_dec=0.999999 --minval=-1 --maxval=1 --quiet=1  --calc_error=1 --file_columns=4 --features=3 --last_item=1 --user_file=user_profile.txt
./toolkits/collaborative_filtering/gensgd --training=rec_log_train.txt --val_pos=2 --rehash=1 --limit_rating=1000000 --max_iter=100 --gensgd_mult_dec=0.999999 --minval=-1 --maxval=1 --quiet=0 --features=3 --last_item=1   --quiet=1 --user_file=user_profile.txt --item_file=item.txt --gensgd_rate5=1e-5 --calc_error=1 --file_columns=4
./toolkits/collaborative_filtering/sparse_gensgd --training=kddb --cutoff=0.5 --calc_error=1 --quiet=1 --gensgd_mult_dec=0.99999 --max_iter=100 --validation=kddb.t --gensgd_rate3=1e-4 --D=20 --gensgd_regw=1e-4 --gensgd_regv=1e-4 --gensgd_rate1=1e-4 --gensgd_rate2=1e-4 --gensgd_reg0=1e-3
