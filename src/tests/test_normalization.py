import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.normalization import WordMatcher

# Global threshold for similarity
THRESHOLD = 0.3

class TestNormalizationMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the WordMatcher instance for all tests."""
        cls.matcher = WordMatcher(threshold=THRESHOLD)

    def test_correct_guess(self):
        """Test if the function correctly guesses the word in the list."""
        word = "Apple"
        word_list = ["Apple Inc.", "Microsoft", "Google", "Amazon"]
        match, _ = self.matcher.match(word, word_list)
        self.assertEqual(match, "Apple Inc.")

    def test_no_match(self):
        """Test if the function correctly determines that no word matches."""
        word = "NonexistentWord"
        word_list = ["Apple", "Banana", "Cherry", "Date"]
        match, _ = self.matcher.match(word, word_list)
        self.assertIsNone(match)

    def test_partial_match(self):
        """Test if the function correctly guesses a partial match."""
        word = "extra large Peaches"
        word_list = ["Peaches", "Apples", "Bananas", "Oranges"]
        match, _ = self.matcher.match(word, word_list)
        self.assertEqual(match, "Peaches")

    def test_misspelling(self):
        """Test if the function correctly guesses a misspelled word."""
        word = "Gooogle"
        word_list = ["Google", "Facebook", "Twitter", "LinkedIn"]
        match, _ = self.matcher.match(word, word_list)
        self.assertEqual(match, "Google")

    def test_similar_phrases(self):
        """Test if the function correctly guesses the best match among similar phrases."""
        word = "Peach box"
        word_list = ["A box of Peaches", "Peach crate", "Peach basket", "Apple box"]
        match, _ = self.matcher.match(word, word_list)
        self.assertEqual(match, "A box of Peaches")

    def test_with_numbers(self):
        """Test if the function correctly guesses phrases with numbers."""
        word = "Version 2.0"
        word_list = ["Version 1.0", "Version 2.0", "Version 3.0"]
        match, _ = self.matcher.match(word, word_list)
        self.assertEqual(match, "Version 2.0")

    def test_with_special_characters(self):
        """Test if the function correctly guesses phrases with special characters."""
        word = "C++"
        word_list = ["Python", "Java", "C++", "JavaScript"]
        match, _ = self.matcher.match(word, word_list)
        self.assertEqual(match, "C++")

if __name__ == '__main__':
    # Run the tests and display a summary of the results
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNormalizationMethods)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\nSummary of Results:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures or result.errors:
        print("\nDetails of Failures and Errors:")
        for failure in result.failures:
            print(failure)
        for error in result.errors:
            print(error)