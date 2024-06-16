STEP=$1
grep -rnw LGT5[1,2]*.log -e "step $STEP:"
