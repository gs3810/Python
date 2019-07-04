from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer, Metadata, Interpreter
from rasa_nlu import config

# loading the nlu training samples
training_data = load_data('Data/nlu_data.md')

# trainer to educate our pipeline
trainer = Trainer(config.load('config.json'))

# train the model!
interpreter = trainer.train(training_data)

# store it for future use
model_directory = trainer.persist("./Models/nlu", fixed_model_name="current")

nlu_model = Interpreter.load('./Models/nlu/default/current')
out = nlu_model.parse('Hello')

print (out)

