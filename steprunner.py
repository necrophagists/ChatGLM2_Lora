from accelerate import Accelerator

class StepRunner:
    def __init__(self,model,loss_fn,accelerator=None,stage="train",
                 metrics_dict=None,optimizer=None,lr_scheduler=None):
        self.model,self.loss_fn,self.metrics_dict,self.stage =model,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler =optimizer,lr_scheduler
        self.accelerator =accelerator if accelerator is not None else Accelerator()

        if self.stage =="train":
            self.model.train()
        else:
            self.model.eval()

    def __call__(self, batch):

        with self.accelerator.autocast():  #混合精度
            loss =self.model(input_ids=batch['input_ids'],labels =batch['labels']).loss
        if self.optimizer is not None and self.stage =="train":
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients ==True: #如果多卡就裁剪梯度
               self.accelerator.clip_grad_norm_(self.model.parameters(),1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        #多卡/单卡
        all_loss =self.accelerator.gather(loss).sum()
        step_losses ={self.stage+"_loss":all_loss.item()}

        step_metrics={}
        if self.stage =="train":
            if self.optimizer is not None:
                step_metrics['lr'] =self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] =0.0

        return step_losses,step_metrics  #返回每一步的loss和学习率

    def save_ckpt_ddp(self,save_path,accelerator=None):
        unwrap_model =accelerator.unwrap_model(self.model)  #ddp时去掉多余的module字段
        unwrap_model.save_pretrained(save_path)

    def save_ckpt(self,save_path,accelerator=None):
        self.model.save_pretrained(save_path)

    def load_ckpt(self,ckpt_path):
        self.model =self.model.from_pretrained(ckpt_path)
        self.from_scratch =False   #False 不从头开始训-->使用预训练的权重