/*
Valid Palindrome
Easy

Given a string s, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
*/

/**
 * @param {string} s
 * @return {boolean}
 */
var isPalindrome = function (s) {
    s = s.toLowerCase();
    s = s.replace(/[\W_]/gim, "");
    rev_s = s.split("").reverse().join("");
    return s == rev_s;
};
