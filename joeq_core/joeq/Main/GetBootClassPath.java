package Main;

public abstract class GetBootClassPath {
    public static void main (String args[]) {
        System.out.println(System.getProperty("sun.boot.class.path"));
        //System.out.println(System.getProperty("sun.boot.class.path")+System.getProperty("path.separator")+System.getProperty("java.class.path"));
    }
}
