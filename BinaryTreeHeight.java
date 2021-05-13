/*
 Write code to calculate the height of a binary tree.
 */
public class BinaryTreeHeight {
    static class Node {
        int data;
        Node left;
        Node right;

        Node(int d) {
            data = d;
            left = null;
            right = null;
        }
    }

    int height(Node root) {
        if (root == null) {
            return 0;
        }
        int lh = height(root.left);
        int rh = height(root.right);

        return (1 + Math.max(lh, rh));

    }

    public static void main(String[] args) {
        BinaryTreeHeight m = new BinaryTreeHeight();
        Node root = new Node(1);
        root.right = new Node(3);
        root.left = new Node(2);
        root.left.left = new Node(4);
        System.out.println(m.height(root));
    }
}