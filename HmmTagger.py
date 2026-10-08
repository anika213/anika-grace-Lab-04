#!/usr/bin/env python3
import argparse
from collections import defaultdict, Counter
import os
import pickle
import sys

from numpy import argmax, zeros, array, float32, ones, zeros, log
import spacy

import read_tags


class HMMTagger():
    def __init__(self, nlp, alpha=0.1):
        '''
        
        Initialize the HMM tagger with ptb_tag_set from read_tags module. This represents the set of all possible tags and all the ways we can call it. Vocabulary is initiallised with an OOV token. Alpha represents a smoothing parameter passed into HMMTagger.
        '''
        self.tags = ["<<START>>",] + read_tags.ptb_tag_set  # ptb_tag_set is all the values (NOT keys) in the lists in read_tags; tehse give more granularity into parts of speeech
        self.vocab = ["<<OOV>>",]
        self.alpha = alpha 

    def clean_token(self, text):
        """Convert each token to lower-case text or a special OOV token
        for out of vocabulary words"""
        return text.lower() if text.lower() in self.vocab else "<<OOV>>"

    def word_to_index(self, w):
        """Given a token, find its index in the vocabulary."""
        return self.vocab.index(self.clean_token(w))

    def tag_to_index(self, t):
        """Given a tag, find its index in the tag list."""
        return self.tags.index(t)

    def update_vocab(self, train_dir):
        """Given a directory of files, populate the vocabulary that corresponds
        to all tokens in the files in the directory."""
        token_counter = Counter()
        for words, _ in read_tags.parse_dir(train_dir): 
            token_counter.update([w.lower() for w in words])
        self.vocab += list(token_counter.keys())
    
    def normalize_probabilities(self):
        """Normalize the tag-word and tag-tag probability matrices
        in log space."""
        self.tag_word_probs = self.normalize(self.tag_word)
        self.tag_tag_probs = self.normalize(self.tag_tag)

    def __call__(self, tokens):
        """If invoked as a function call, predicts tags for a list
        of tokens given the trained model."""
        self.predict(tokens)
        
    ## BEGIN DOCUMENTATION HERE ###

    def get_start_costs(self):
        """This method returns the start costs for the HMM, which are the transition probabilities from the start tag to all other tags."""
        return self.tag_tag_probs[self.tag_to_index("<<START>>"),:] # probability to transition from start tag to all other tagsl qw get the index of start so we see which row it is in the matrix, then copy that row

    def get_token_costs(self, token):
        """
        This method returns the token costs for a given token, which are the emission probabilities for that token across all tags.
        """
        return self.tag_word_probs[:,self.word_to_index(token.text)] 
    
    def normalize(self, m):
        """This function takes is a vector and normalizes it in log space: trasnspose it so in the end, our vector sums to 1."""
        
        return (log(m).transpose() - log(m.sum(axis=1))).transpose()

    def do_train_sent(self, words, tags): 
        """
        Adds counts for a given set of words with tags (occurances of the words) to the matrices which will be later used to predict/classify
        
        """
        # tag_word has rows that are tagm columns that are word
        # so tag_word [t_i][w_i] = P(tag | word)
        prev_tag = self.tag_to_index("<<START>>")
        for word, tag in zip(words, tags):
            t_i = self.tag_to_index(tag)
            w_i = self.word_to_index(word)
            self.tag_word[t_i][w_i] += 1 # add count to emission matrix (so we know how many times the word appeared with this tag)
            self.tag_tag[prev_tag][t_i] += 1 # add count to transition matrix (so we know how many times the last state came before this one)
            prev_tag = t_i

    def train(self, train_dir):
        """TODO: This is the main function which initializes, and then populates the HMM matrices with the counts (via do_train_sent) with training data for DP to work. It also normalizes to convert everything to the log space."""
        self.update_vocab(train_dir)


        # len(self.tags) = N
        # len(self.vocab) = V
        self.tag_word  = ones((len(self.tags), len(self.vocab))) * self.alpha # emission, ones is the all ones matrix: inits a matrix of size N x V
        self.tag_tag =  ones((len(self.tags), len(self.tags))) * self.alpha # transition from tag to tag, matrix size N x N
        # sets initial values of these matrices to alpha for alpha smoothing (so if we have no tag-tag or tag-word occurences, it seems rare but nonzero)

        for words, tags in read_tags.parse_dir(train_dir):
            self.do_train_sent(words, tags)
        self.normalize_probabilities() # log normalize probabilities

    def predict(self, tokens):
        """This is the main Viterbi algorithm implemenation which does DP using the probabilties that we populated in the train function. """
        # tagtag = A
        # wordtag = B
        
        
        # Build DP table, which should be |sent| x |tags|
        cost_table = zeros((len(tokens), len(self.tags)), float32) # initializes costs to 0  
        bt_table   = zeros((len(tokens), len(self.tags)), int)
        # initialize backtrace table to zeroes.

        for token_i, token in enumerate(tokens):
            token_costs = self.get_token_costs(token) # get emission probabilities for this token across tags
            if token_i == 0: 
                cost_table[token_i, :] = self.get_start_costs() + token_costs # base case, if we are on the zeroth token we multiply P(part of speech | start of sent) * P(token | start of speech); addition is multiplication on log space
                bt_table[token_i, :] = -1 # notate end of sequence
            else:
                costs = self.tag_tag_probs.copy() 
                # TODO: Fill in the actual costs matrix to compute
                # the sum of the log probability from the last state,
                # the transition log probability,
                # and the emission log probability
            
                # add B + A + D such that we add B_j, word_index + A_j' (rows that we max over), j + D_j', j

                word_index = word_to_index(token)
                # A = tag_tag = rows are j', cols are j
                # B = tag_word_probs, rows are j, cols are word

                costs = A + B[:, word_index] + cost_table[token_i - 1, :]

                # j' = last tag possibilities is rows of A
                # j = current iteration tag is cols of A
                # because B indexes on current iteration tag
                # we want B to change over cols
                # so B should be a row 
                
                cost_table[token_i, :] = costs.max(axis=0) # gets the acutal max costs 
                bt_table[token_i,:] = costs.argmax(axis=0) # gets the actual arguments which maximise the costs

    
        # Find the highest-probability tag for last word
        best_last_tag = argmax(cost_table[token_i, :])

        # Trace back through the breadcrumb table
        self.backtrace(bt_table, tokens, best_last_tag)

    def backtrace(self, bt_table, tokens, best_last_tag):
        """We backtrace through the cost matrix to reconstruct the optimal tag sequence based on the maximum probabilities which were computed during the forward pass."""
        current_row = len(tokens)-1
        for t in list(tokens)[::-1]:
            t.tag_ = self.tags[best_last_tag]
            best_last_tag = bt_table[current_row, best_last_tag]
            current_row -= 1


def main(args):
    nlp = spacy.load("en_core_web_sm")
    tagger = HMMTagger(nlp, alpha=args.alpha)
    tagger.train(args.dir)
    pickle.dump(tagger, args.output)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Train (and save) hmm models for POS tagging')
    parser.add_argument("--dir", "-d", metavar="DIR", required=True,
                        help="Read training data from DIR")
    parser.add_argument("--output", "-o", metavar="FILE", 
                        type=argparse.FileType('wb'), required=True,
                        help="Save output to FILE")
    parser.add_argument("--alpha", "-a", default=0.1, 
                        help="Alpha value for add-alpha smoothing")

    args = parser.parse_args()
    main(args)
