import collections
from turtle import st

import numpy as np

import util


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***

    words = message.split()
    words = [word.lower() for word in words]

    return words

    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***

    word_count = collections.defaultdict(int)

    for message in messages:
        words = get_words(message)
        for word in words:
            word_count[word] += 1
    
    dictionary = collections.defaultdict(int)
    index = 0

    for word, count in word_count.items():
        if count >= 5:
            dictionary[word] = index
            index += 1
    
    return dictionary

    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***

    matrix = np.zeros((len(messages), len(word_dictionary))) # where i is the message index and j is the index of the word in the dictionary

    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                matrix[i, word_dictionary[word]] += 1
    
    return matrix

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    X_spam = matrix[labels == 1]
    X_ham = matrix[labels == 0]

    laplace_alpha = 1

    word_counts_spam = X_spam.sum(axis=0)
    word_counts_ham = X_ham.sum(axis=0)

    total_spam_words = word_counts_spam.sum()
    total_ham_words = word_counts_ham.sum()

    # Probability of getting a word w given that the email is spam is the count of w in spam emails + alpha (Laplace smoothing) / total spam words + alpha * voab size (since we added an additional count for each word)
    p_word_given_spam = (word_counts_spam + laplace_alpha) / (total_spam_words + laplace_alpha * matrix.shape[1])
    p_word_given_ham = (word_counts_ham + laplace_alpha) / (total_ham_words + laplace_alpha * matrix.shape[1])

    phi = np.mean(labels)
    log_phi = np.log(phi)
    log_phi_ham = np.log(1 - phi)

    log_p_word_given_spam = np.log(p_word_given_spam)
    log_p_word_given_ham = np.log(p_word_given_ham)

    model = {
        "log_phi": log_phi,
        "log_phi_ham": log_phi_ham,
        "log_p_word_given_spam": log_p_word_given_spam,
        "log_p_word_given_ham": log_p_word_given_ham,
    }

    return model

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***

    log_phi = model["log_phi"]
    log_phi_ham = model["log_phi_ham"]
    log_p_word_given_spam = model["log_p_word_given_spam"]
    log_p_word_given_ham = model["log_p_word_given_ham"]

    log_scores_spam = log_phi + matrix @ log_p_word_given_spam
    log_scores_ham = log_phi_ham + matrix @ log_p_word_given_ham

    y_pred = (log_scores_spam > log_scores_ham).astype(int)

    return y_pred

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    
    log_p_word_given_spam = model["log_p_word_given_spam"]
    log_p_word_given_ham = model["log_p_word_given_ham"]

    log_ratios = log_p_word_given_spam - log_p_word_given_ham

    top_5_indices = np.argsort(log_ratios)[::-1][:5]

    index_to_word = collections.defaultdict(str)

    for word, index in dictionary.items():
        index_to_word[index] = word

    top_5_words = []

    for index in top_5_indices:
        top_5_words.append(index_to_word[index])

    return top_5_words
    
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

if __name__ == "__main__":
    main()
