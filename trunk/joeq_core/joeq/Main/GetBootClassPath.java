package Main;

public abstract class GetBootClassPath {
    public static void main (String args[]) {
        for (int i=0; i<args.length; ++i) {
            System.out.print(args[i]+System.getProperty("path.separator"));
        }
        System.out.print(System.getProperty("sun.boot.class.path"));
        //System.out.print(System.getProperty("path.separator")+System.getProperty("java.class.path"));
    }
}
