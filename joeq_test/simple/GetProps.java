package simple;
public abstract class GetProps {
    public static void main (String args[]) {
        for (java.util.Iterator i=System.getProperties().entrySet().iterator();
             i.hasNext(); ) {
            System.out.println(i.next());
        }
    }
}
