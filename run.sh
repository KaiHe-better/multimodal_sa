nohup python main.py --dataset mosi --save_results  --gpu 0  > /dev/null 2>&1 &   
nohup python main.py --dataset mosei --save_results --gpu 1  > /dev/null 2>&1 &   
nohup python main.py --dataset chsims --save_results --gpu 3 > /dev/null 2>&1 &   



nohup python main.py --dataset mosi --save_results  --gpu 5  > /dev/null 2>&1 &   
nohup python main.py --dataset mosei --save_results --gpu 6  > /dev/null 2>&1 &   
nohup python main.py --dataset chsims --save_results --gpu 7 > /dev/null 2>&1 &   