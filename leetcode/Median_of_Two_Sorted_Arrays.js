/*
Median of Two Sorted Arrays
Hard

Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).
*/

/**
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number}
 */
var findMedianSortedArrays = function (nums1, nums2) {
    let newArr = [...nums1, ...nums2];
    newArr.sort((a, b) => {
        return a - b;
    });

    //     if (nums1.length == 0 && nums2.length != 0) {
    //         newArr = nums2;
    //     }

    //     if (nums2.length == 0 && nums1.length != 0) {
    //         newArr = nums1;
    //     }

    if (newArr.length % 2 != 0) {
        return newArr[Math.floor(newArr.length / 2)];
    } else {
        return (newArr[newArr.length / 2] + newArr[newArr.length / 2 - 1]) / 2;
    }
};
