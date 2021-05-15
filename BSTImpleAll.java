/*implementation of BST
    insertion 
    seraching 
    printing
    deletion of node
*/
public class BSTImpleAll {
    static Node root;

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

    void addNode(int k) {
        root = addN(root, k);
    }

    static Node addN(Node root, int k) {
        if (root == null) {
            root = new Node(k);
            return root;
        }
        if (root.data < k) {
            root.right = addN(root.right, k);
        } else if (root.data > k) {
            root.left = addN(root.left, k);
        }
        return root;
    }

    Node search(Node root, int val) {
        if (root == null || root.data == val) {
            return root;
        } else if (val > root.data) {
            return search(root.right, val);
        }
        return search(root.left, val);
    }

    static void inorder(Node root) {
        if (root == null) {
            return;
        }
        inorder(root.left);
        System.out.print(root.data + " ");
        inorder(root.right);
    }

    Node remove(Node root, int val) {
        if (root == null) {
            return root;
        } else if (val > root.data) {
            root.right = remove(root.right, val);
        } else if (val < root.data) {
            root.left = remove(root.left, val);
        } else { // if found the val
                 // case 1 : if there is no child
            if (root.left == null && root.right == null) {
                root = null;
            } // case 2: if only one child
            if (root.left == null) {
                return root.right;
            } else if (root.right == null) {
                return root.left;
            }
            // case 3 : if has two childs
            root.data = minval(root.right);

            root.right = remove(root.right, root.data);

        }
        return root;
    }

    int minval(Node root) {
        int minv = root.data;
        while (root.left != null) {
            minv = root.left.data;
            root = root.left;
        }
        return minv;
    }

    public static void main(String[] args) {
        BSTImpleAll m = new BSTImpleAll();
        m.addNode(6);
        m.addNode(3);
        m.addNode(9);
        m.addNode(1);
        m.addNode(7);
        m.addNode(3);
        System.out.println("the inorder traversal is :");
        inorder(root);
        System.out.println("\nthe element is found");
        System.out.println(m.search(root, 6).data);
        m.remove(root, 3);
        System.out.println("after performing deletion:");
        inorder(root);
    }

}
