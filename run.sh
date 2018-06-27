python main.py -l en -tb English -t parse \
	--build_data False --train True --test False \
	--enable_elmo True --enable_seqlabel False \
	--embedd_keep_prob 0.67 \
	--encoder transformer --birnn_nlayer 2 \
	--transformer_nlayer 2 --nhead 5 --model_size 300 --pos_hidden_size 1024 \
	--clip 5.0 --batch_size 16 --device_ids 2-3-4
