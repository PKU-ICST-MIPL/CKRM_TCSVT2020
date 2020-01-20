def read_log(log):
    log_file = open(log,'r')
    Flag = False
    print("Epoch","train_acc","train_loss","val_acc","val_loss")
    for line in log_file:
        elements = line.split()
        if "TRAIN EPOCH" in line:
            Flag = True
        if Flag == True and "loss"==line.split()[0]:
            train_loss = round(float(line.split()[1]),3)
        if Flag == True and "accuracy" in line:
            train_acc = round(float(line.split()[1]),3)
            Flag = False
        if "Val epoch" in line:
            epoch = str(elements[2])
            acc = str(elements[5])
            loss = str(elements[8])
            Flag = False
            print(epoch,train_acc,train_loss,acc,loss)
    log_file.close()

print("The Q->A training results for each epoch:")
read_log("./saves/flagship_answer/stdout.log")
print("The QA->R trainging results for each epoch:")
read_log("./saves/flagship_rationale/stdout.log")

