/*
String Matching in an Array
Easy

Given an array of string words. Return all strings in words which is substring of another word in any order. 

String words[i] is substring of words[j], if can be obtained removing some characters to left and/or right side of words[j].
*/

/**
 * @param {string[]} words
 * @return {string[]}
 */
var stringMatching = function (words) {
    res = [];
    for (word1 of words) {
        const re = new RegExp(word1, "gim");
        for (word2 of words) {
            if (word1 != word2 && re.test(word2) && !res.includes(word1)) {
                res.push(word1);
            }
        }
    }
    return res;
};
