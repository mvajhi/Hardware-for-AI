M_PATH='./model'
R_PATH='./results'

if [ ! -d "$R_PATH" ]; then
    mkdir $R_PATH
fi

for i in $(ls $M_PATH/*.c); do
    echo "####Testing $i####"
    make clean && make runner MODEL_C_FILE=$i 2> /dev/null && { time ./runner; } 2>&1 | tee $R_PATH/$(basename $i .c).log
    gprof runner gmon.out >> $R_PATH/$(basename $i .c).log
done

rm runner gmon.out

echo "####Testing all models done####"