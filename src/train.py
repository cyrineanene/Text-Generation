from transformers import Trainer, TrainingArguments
from src.load_data import load_dataset, load_data_collator


#Step 2: Train the model
def train(model, tokenizer, output_dir, train_dataset, data_collator, overwrite_output_dir = False, per_device_train_batch_size = 8, num_train_epochs = 50.0, save_steps = 50000):
    
    training_args = TrainingArguments(
          output_dir=output_dir,
          overwrite_output_dir=overwrite_output_dir,
          per_device_train_batch_size=per_device_train_batch_size,
          num_train_epochs=num_train_epochs,
          save_steps=save_steps,
      )

    trainer = Trainer(
          model=model,
          args=training_args,
          data_collator=data_collator,
          train_dataset=train_dataset,
    )
      
    trainer.train()

    model.save_pretrained('./pretrained/model')
    tokenizer.save_pretrained('./pretrained/tokenizer')

