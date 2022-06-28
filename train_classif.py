import torch
import random
from torchtext.legacy import data
from torchtext.legacy import datasets
from utils.model import RNN_classif
from utils.helper import binary_accuracy
import torch.optim as optim

#
# GLOBAL VARIABLES
#

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm')
LABEL = data.LabelField(dtype = torch.float)



#
# Data preparation
#



# Load the IMDB dataset (see datasets from torchtext.legacy)
# Split the train using .split() to obtain a valid set.
# Print the statistics of the train, val, test for monitoring
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(vars(train_data.examples[0]))

train_data, valid_data = train_data.split(random_state = random.seed(SEED))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')



#
# Vocabulary building (as we are not using w2v here)
#



#
# Here, we can use the .build_vocab() of Field and LabelField.
# Note that we will need to define a maximum size for the vocab creation
# It's up to you to find a good number ..
#

# DO YOUR STUFF

# Once done, the vocab info can be obtained with
print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

# Most common
print(TEXT.vocab.freqs.most_common(20))

# Verify that the label are ok as well (.vocab.stoi())



#
# Data iterator and bucket preparation
#

#
# We'll use a BucketIterator which is a special type of iterator ... I'll let you
# Find why it is cool in our case :p
#

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = ...


#
# Model initialisation
#



model = RNN(...)
model = model.to(device)

#
# For the fun, we get the number of trainable parameters in our model
#
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('The model has '+str(num_params)+' trainable parameters')



#
# Training inisialisation
#

optimizer =
criterion = (...).to(device)

# Go to utils/helper to implement the accuracy function !

#
# Let's train !
#

N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'weights/RNN_classif-model.pt')

    # Print your stuff
    print(...)
