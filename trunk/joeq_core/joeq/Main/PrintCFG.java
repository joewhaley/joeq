package Main;

import Clazz.jq_Class;

public class PrintCFG {
    public static void main(String[] args) {
	jq_Class[] c = new jq_Class[args.length];
	for (int i = 0; i < args.length; i++) {
	    c[i] = Helper.load(args[i]);
	}

	Compil3r.Quad.PrintCFG pass = new Compil3r.Quad.PrintCFG();

	for (int i = 0; i < args.length; i++) {
	    Helper.runPass(c[i], pass);
	}
    }
}
