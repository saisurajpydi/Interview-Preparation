import java.util.*;
import java.lang.*;

public class KthSmallInArray {
    public static int partition(int arr[], int low, int high) {
        int pivot = arr[high];
        int prev = low;
        for (int j = low; j <= high; j++) {
            if (arr[j] < pivot) {
                int temp = arr[j];
                arr[j] = arr[prev];
                arr[prev] = temp;
                prev += 1;
            }
        }
        int temp = arr[prev];
        arr[prev] = arr[high];
        arr[high] = temp;

        return prev;
    }

    public static int quickSelect(int arr[], int low, int high, int k) {
        int parti = partition(arr, low, high);
        if (parti == k) {
            return arr[parti];
        } else if (parti > k) {
            return quickSelect(arr, 0, parti - 1, k);
        } else {
            return quickSelect(arr, parti + 1, high, k);
        }
    }

    public static void main(String[] args) {
        KthSmallInArray m = new KthSmallInArray();
        int arr[] = new int[] { 3, 7, 4, 2, 8, 1 }; // {10, 4, 5, 8, 6, 11, 26};
        int length = arr.length;
        int k = 4;
        int high = length - 1;
        System.out.println(k + "th smallest No is: " + quickSelect(arr, 0, high, k - 1));
    }

}
