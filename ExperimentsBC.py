from attention.configurations import generate_config
from attention.Trainers.TrainerBC import Trainer, Evaluator
            
def train_dataset(dataset, args, config='lstm') :
        config = generate_config(dataset, args, config)
        trainer = Trainer(dataset, args, config=config)
        #go ahead and save model
        dirname = trainer.model.save_values(save_model=False)
        print("DIRECTORY:", dirname)
        if args.adversarial :
            trainer.train_adversarial(dataset.train_data, dataset.test_data, args)
        else :
            trainer.train_standard(dataset.train_data, dataset.test_data, args, save_on_metric=dataset.save_on_metric)
        print('####################################')
        print("TEST RESULTS FROM BEST MODEL")
        evaluator = Evaluator(dataset, trainer.model.dirname, args)
        _ = evaluator.evaluate(dataset.test_data, save_results=True)
        return trainer, evaluator

def train_dataset_on_encoders(dataset, args, exp_name) :
	train_dataset(dataset, args, exp_name)
