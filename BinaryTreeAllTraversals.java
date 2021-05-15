import java.util.*;

public class BinaryTreeAllTraversals {
    static class Node {
        int key;
        Node left;
        Node right;

        Node(int k) {
            key = k;
            left = null;
            right = null;
        }
    }

    void printPostorder(Node node) {
        if (node == null)
            return;

        printPostorder(node.left);

        printPostorder(node.right);
        System.out.print(node.key + " ");
    }

    void printInorder(Node node) {
        if (node == null)
            return;
        printInorder(node.left);
        System.out.print(node.key + " ");
        printInorder(node.right);
    }

    void printPreorder(Node node) {
        if (node == null)
            return;
        System.out.print(node.key + " ");
        printPreorder(node.left);
        printPreorder(node.right);
    }

    void printLevelOrder(Node root) {
        System.out.println("\nthe level order is :");
        ArrayList<Node> alist = new ArrayList<>();
        alist.add(root);
        while (!alist.isEmpty()) {
            root = alist.get(0);
            alist.remove(0);
            System.out.print(root.key + " ");
            if (root.left != null) {
                alist.add(root.left);
            }
            if (root.right != null) {
                alist.add(root.right);
            }
        }
    }

    public static void main(String[] args) {
        BinaryTreeAllTraversals tree = new BinaryTreeAllTraversals();
        Node root = new Node(1);
        root.left = new Node(2);
        root.right = new Node(3);
        root.left.left = new Node(4);
        root.left.right = new Node(5);

        System.out.println("Preorder traversal of binary tree is ");
        tree.printPreorder(root);

        System.out.println("\nInorder traversal of binary tree is ");
        tree.printInorder(root);

        System.out.println("\nPostorder traversal of binary tree is ");
        tree.printPostorder(root);

        tree.printLevelOrder(root);
    }
}