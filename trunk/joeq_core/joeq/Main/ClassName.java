/*
 * Created on Oct 24, 2003
 *
 * To change the template for this generated file go to
 * Window - Preferences - Java - Code Generation - Code and Comments
 */
package Main;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;

import Bootstrap.PrimordialClassLoader;
import Clazz.jq_Class;
import Clazz.jq_ClassFileConstants;
import Clazz.jq_ConstantPool;

/**
 * @author jwhaley
 *
 * To change the template for this generated type comment go to
 * Window - Preferences - Java - Code Generation - Code and Comments
 */
public class ClassName implements jq_ClassFileConstants {

    public static void main(String[] args) throws Exception {
        HostedVM.initialize();
        
        for (int i = 0; i < args.length; ++i) {
            try {
                System.err.println(args[i]);
                DataInputStream in = new DataInputStream(new FileInputStream(args[i]));
                int k = in.skipBytes(8);
                if (k != 8) throw new IOException();
                int constant_pool_count = in.readUnsignedShort();
                jq_ConstantPool cp = new jq_ConstantPool(constant_pool_count);
                cp.load(in);
                cp.resolve(PrimordialClassLoader.loader);
                k = in.skipBytes(2);
                if (k != 2) throw new IOException();
                char selfindex = (char)in.readUnsignedShort();
                if (cp.getTag(selfindex) != CONSTANT_ResolvedClass) {
                    System.err.println("constant pool entry "+(int)selfindex+", referred to by field this_class" +
                                       ", is wrong type tag (expected="+CONSTANT_Class+", actual="+cp.getTag(selfindex)+")");
                }
                jq_Class t = (jq_Class) cp.get(selfindex);
                System.out.println(t.getJDKName());
                in.close();
            } catch (Exception x) {
                x.printStackTrace();
            }
        }
    }
}
