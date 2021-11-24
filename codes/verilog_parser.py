import os
import collections

from typing import List, Dict, Tuple, Optional
import pyverilog
from pyverilog.vparser.parser import parse
import re

import networkx as nx

import cProfile

def parse_arg(arg,port_info,ios,wires):

    if type(arg) == pyverilog.vparser.ast.Identifier:
        if wires.get(arg.name,None) is not None:
            high_bit, low_bit = wires[arg.name]
        elif ios.get(arg.name,None) is not None:
            high_bit, low_bit = ios[arg.name]
            port_info.flag_update = True
            port_info.args_need_update.add(arg.name)
        else:
            assert False
        width = high_bit-low_bit+1
        if width == 1:
            port_info.arg_list.append(arg.name)
        else:
            for i in range(high_bit,low_bit-1,-1):
                port_info.arg_list.append("{}_{}".format(arg,i))
        #arg.show()
    elif type(arg) == pyverilog.vparser.ast.IntConst:
        port_info.arg_list.append(arg.value)
    elif type(arg) == pyverilog.vparser.ast.Partselect:
        arg_nm,high_bit,low_bit = arg.children()
        arg_nm = arg_nm.name
        high_bit, low_bit = int(str(high_bit)),int(str(low_bit))
        if high_bit < low_bit:
            temp = high_bit
            high_bit = low_bit
            low_bit = temp
        for i in range(high_bit,low_bit-1,-1):
            port_info.arg_list.append("{}_{}".format(arg_nm,i))
        if ios.get(arg_nm,None) is not None:
            port_info.flag_update = True
            port_info.args_need_update.add(arg_nm)
    elif type(arg) == pyverilog.vparser.ast.Pointer:
        arg_nm, position = arg.children()
        arg_nm = arg_nm.name
        port_info.arg_list.append("{}_{}".format(arg_nm,position))
        if ios.get(arg_nm,None) is not None:
            port_info.flag_update = True
            port_info.args_need_update.add(arg_nm)
    else:
        print(arg)
        assert False

class MultInfo:
    cell_name:str
    fanins:dict
    fanouts:dict

    def __init__(self,cell_name):
        self.cell_name = cell_name
        self.fanins = {}
        self.fanouts = {}

class ModuleInfo:
    cell_name:str
    cell_type:str
    instance_name:str
    ports:dict
    index:int
    def __init__(self,cell_name,cell_type,instance_name):
        self.cell_name = cell_name
        self.cell_type = cell_type
        self.instance_name = instance_name
        self.ports = {}
        self.index = -1
class PortInfo:
    ptype:str
    portname: str
    argname: str
    argcomp: str
    is_adder_input: bool
    is_adder_output: bool
    is_sub_input1: bool
    is_sub_input2: bool
    is_muldiv_output:bool
    is_muldiv_input1: bool
    is_muldiv_input2: bool
    is_sub_output: bool
    input_comp: str
    output_comp: str
    arg_list: list
    position:tuple
    flag_update:bool
    args_need_update:set
    flag_mult:bool
    def __init__(self, portname,argname, argcomp):
        self.ptype = None
        self.portname = portname
        self.argname = argname
        self.argcomp = argcomp
        self.is_adder_input = False
        self.is_adder_output = False
        self.is_sub_input1 = False
        self.is_sub_input2 = False
        self.is_muldiv_output =  False
        self.is_muldiv_input1 = False
        self.is_muldiv_input2 = False
        self.is_sub_output = False
        self.arg_list = []
        self.position = None
        self.flag_update = False
        self.args_need_update = set()
        self.flag_mult = False
class DcParser:
    def __init__(
        self, top_module: str,adder_keywords: List[str], sub_keywords: List[str], hadd_type: str = "hadd"
    ):
        self.top_module = top_module
        self.adder_keywords = adder_keywords
        self.sub_keywords = sub_keywords
        assert hadd_type in ("hadd", "hadd_s", "xor")
        self.hadd_type = hadd_type  # treat hadd sum as either "hadd", "hadd_s", "xor"
        self.hadd_name_dict = {}
        if hadd_type == "hadd":
            self.hadd_name_dict["hadd_s"] = "HADD"
            self.hadd_name_dict["hadd_c"] = "HADD"
            self.hadd_name_dict["fadd_s"] = "FADD"
            self.hadd_name_dict["fadd_c"] = "FADD"
        elif hadd_type == "hadd_s":
            self.hadd_name_dict["hadd_s"] = "HADD_S"
            self.hadd_name_dict["hadd_c"] = "HADD_C"
            self.hadd_name_dict["fadd_s"] = "FADD_S"
            self.hadd_name_dict["fadd_c"] = "FADD_C"
        elif hadd_type == "xor":
            self.hadd_name_dict["hadd_s"] = "XOR"
            self.hadd_name_dict["hadd_c"] = "AND"
            self.hadd_name_dict["fadd_s"] = "XOR"
            self.hadd_name_dict["fadd_c"] = "MAJ"
        self.muldivs = []
    def is_input_port(self, port: str) -> bool:
        return not self.is_output_port(port)

    def is_output_port(self, port: str) -> bool:
        return port in ("Y", "S", "SO", "CO", "C1", "Q", "QN")

    def parse_report(self,fname):
        with open(fname,'r') as f:
            text = f.read()
        cells  = text.split('Datapath Report for')
        print('number of cells',len(cells))
        cells = cells[1:]
        dp_target_cells = {}

        for cell in cells:
            cell = cell.split('Implementation Report')[0]
            cell = cell[:cell.rfind('\n==============================================================================')]
            cell_name = cell.split('\n')[0].replace(' ','')
            vars = cell[cell.rfind('=============================================================================='):]
            vars = vars.split('\n')[1:]
            #print(cell_name,len(vars))
            var_types = {}
            flag_mul = False
            for var in vars:
                var = var.replace(' ','')
                _,var_name,type,data_class,width,expression,_ =var.split('|')
                var_types[var_name] = (type)
                #print(var_name,type,width,expression)
                if '*' in expression :
                    flag_mul = True
                    self.muldivs.append(cell_name)
                    dp_target_cells[cell_name] = dp_target_cells.get(cell_name, ( 'muldiv',{}, {}) )
                    inputs = expression.split('*')
                    for input in inputs:
                        dp_target_cells[cell_name][1][input] = 2
                    #print(adder_cells)
                if '+' in expression and '-' not in expression:
                    #print(var_name, type, width, expression)
                    #print(adder_cells)
                    dp_target_cells[cell_name] = dp_target_cells.get(cell_name, ('add', {}, {}))
                    dp_target_cells[cell_name][2][var_name] = 1
                    inputs = expression.split('+')
                    for input in inputs:
                        dp_target_cells[cell_name][1][input] = 1
                if '-' in expression and '+' not in expression:
                    dp_target_cells[cell_name] = dp_target_cells.get(cell_name, ('sub', {}, {}))
                    dp_target_cells[cell_name][2][var_name] = 1
                    inputs = expression.split('-')
                    for i,input in enumerate(inputs):
                        dp_target_cells[cell_name][1][input] = 1 if i==0 else 2
                    #print(adder_cells)
        print('dp_target_cells',dp_target_cells)
                    #adder_cells[cell_name][0]
                #print(var)
            #print(vars)

        return dp_target_cells
    def parse_port_hier(
        self, ios:dict,wires:dict, port: pyverilog.vparser.parser.Portlist,
    ) -> PortInfo:
        #print(dir(port))
        portname, argname = port.portname, port.argname
        port_info = PortInfo(portname,None,None)
        
        if type(argname) == pyverilog.vparser.ast.Concat:
            args = argname.children()
            for arg in args:
                parse_arg(arg,port_info,ios,wires)
        else:
            parse_arg(argname,port_info,ios,wires)

        return port_info

    def parse_port(
        self, mcomp: str,target_cells: list,port: pyverilog.vparser.parser.Portlist,index01:list,dp_inputs:list,dp_outputs:list
    ) -> PortInfo:
        portname, argname = port.portname, port.argname
        if type(argname) == pyverilog.vparser.ast.Partselect:
            print(argname)

        if type(argname) == pyverilog.vparser.ast.Pointer:
            argname = str(argname.var) + "_" + str(argname.ptr)
        elif type(argname) == pyverilog.vparser.ast.IntConst:
            argname = argname.__str__()
        else:  # just to show there could be various types!
            argname = argname.__str__()
        argcomp = argname[: argname.rfind("_")]
        position = None
        mcomp_lower = mcomp.lower()
        #if 'add_x' in mcomp or "alu_DP_OP" in mcomp: print(portname, argname)
        #if 'div_DP_OP' in mcomp : print(portname,argname)
        if argname == "1'b0" :
            argname = "{}_{}".format(argname,index01[0])
            index01[0] += 1
        elif argname =="1'b1":
            argname = "{}_{}".format(argname, index01[1])
            index01[1] += 1

        port_info = PortInfo(portname, argname, argcomp)

        if portname in ("CLK"):  # clock
            port_info.ptype = "CLK"
            return port_info
        elif self.is_output_port(portname):
            port_info.ptype = "fanout"
        else:
            port_info.ptype = "fanin"

        for muldiv in self.muldivs:
            if muldiv in mcomp:
                port_info.flag_mult = True
                break
        is_target = False
        for kw in self.adder_keywords:
            if kw in mcomp :
                is_target = True
                break
        for kw in self.sub_keywords:
            if kw in mcomp :
                is_target = True
                break
        if len(dp_inputs)!=0 or len(dp_outputs)!=0:
            is_target = True
        if is_target and mcomp != argcomp:
            module_ports = None
            # for module in target_cells.keys():
            #     if module in mcomp:
            #         module_ports = target_cells[module]
            #         break
            # for cases that instance_name is not unique, e.g, have several add_x_1，each is instance of different cell,
            # in theses cases, mcomp contains both cell information and instance information
            cell_type = None
            for module_info in target_cells:
                if module_info.instance_name.lower() in mcomp.lower():
                    module_ports = module_info.ports
                    cell_type = module_info.cell_type
                    break
            if module_ports is None:
                print('module_ports is none', mcomp, portname, argname)
                return port_info
            # assert module_ports is not None
            # print(argname,module_ports)
            for mport in module_ports.keys():
                mport_args = module_ports[mport]
                for i, arg in enumerate(mport_args):
                    if arg.lower() in argname.lower():
                        position = (mport, len(mport_args) - 1 - i)
                        break
            # if position is None:
            #     for mport in module_ports.keys():
            #         mport_args = module_ports[mport]
            #         for i, arg in enumerate(mport_args):
            #             sub1 = arg[arg.find('_') + 1:]
            #             sub2 = sub1[sub1.find('_')+1:]
            #             if sub2.lower() in argname.lower():
            #                 position = (mport, len(mport_args) - 1 - i)
            #                 break
            if position is None:
                # print(module_ports)
                # print(module_ports)
                if "1'b0" in argname or "1'b1" in argname:
                    #port_info.ptype = 'fanin'
                    return port_info
                if re.match("n\d+$", argname) is not None:
                    #port_info.ptype = 'wire'
                    return port_info
                print(mcomp)
                print(portname, argname)
                # print(module_ports)
                # position = False
                pos = argname.split('_')[-1]
                if re.match('\d+$',pos) is None:
                    position = ('E',0)
                else:
                    position = ('E', int(pos))
                # print('output', argname)
                # if position > 100:
                #     assert False
                # print(mcomp,module_ports)
            # assert  position is not None
            port_info.position = position

            if self.is_output_port(portname) :
                if len(dp_outputs) != 0 and position[0] not in dp_outputs.keys():
                    return port_info
                # if contain_mult:
                #     port_info.flag_mult = True
                if cell_type == 'add':
                    port_info.is_adder_output = True
                elif cell_type == 'sub':
                    port_info.is_sub_output = True
                elif cell_type == 'muldiv':
                    port_info.is_muldiv_output = True
                else:
                    print(cell_type)
                    assert  False

                port_info.output_comp = mcomp
            else:
                if len(dp_inputs) != 0 and position[0] not in dp_inputs.keys():
                    return port_info
                if cell_type == 'add':
                    port_info.is_adder_input = True

                    # if len(mult_inputs) != 0 and position[0] in mult_inputs:
                    #     #print('mul2', position[0])
                    #     port_info.is_muldiv_input2 = True
                    # elif len(mult_inputs)!=0 and position[0] in key_inputs:
                    #     #print('mul1',position[0])
                    #     port_info.is_muldiv_input1 = True

                elif cell_type == 'sub':
                    if len(dp_inputs)!=0:
                        # if dp_inputs.get(position[0],None) is None:
                        #     print(mcomp,position[0],dp_inputs)
                        sub_position = dp_inputs[position[0]]
                        if sub_position == 1:
                            port_info.is_sub_input1 = True
                        else:
                            port_info.is_sub_input2 = True

                    else:
                        if position[0] == 'A' :
                            port_info.is_sub_input1 = True
                        elif position[0] == 'B' :
                            port_info.is_sub_input2 = True
                        else:
                            print(mcomp,position[0],port_info.portname)
                            return port_info
                elif cell_type == 'muldiv':
                    if len(dp_inputs)!=0:
                        sub_position = dp_inputs[position[0]]
                        if sub_position == 1:
                            port_info.is_muldiv_input1 = True
                        else:
                            port_info.is_muldiv_input2 = True

                    else:
                        if position[0] in ('I1','I2') :
                            port_info.is_muldiv_input2 = True
                        elif position[0] == 'I3' :
                            port_info.is_muldiv_input1 = True
                        else:
                            print(mcomp,position[0],port_info.portname)
                            return port_info
                else:
                    print(cell_type)
                    assert False
                port_info.input_comp = mcomp

        elif is_target and argcomp != mcomp:
            #print(kw, argname, mcomp)
            assert False
            port_info.is_adder_output = True
            port_info.output_comp = argcomp

        #if 'add_x' in mcomp or 'alu_DP_OP' in mcomp: print(position)
        return port_info

    def parse_hier(self, fname,dp_target_cells):
        """ parse dc generated verilog """
        target_cells = {}
        ast, directives = parse([fname])
        args_to_update = {}
        # print(dir(ast))
        # ast.show()
        # print(dir(directives))
        # exit()
        for module in ast.description.definitions:

            ios = {}
            wires = {}
            for sentence in module.children():
                if type(sentence) == pyverilog.vparser.ast.Decl:

                    for decl in sentence.children():
                        name = decl.name
                        if decl.width is None:
                            high_bit, low_bit = 0, 0
                        else:
                            high_bit, low_bit = decl.width.children()
                            high_bit,low_bit = int(high_bit.value),int(low_bit.value)
                            if high_bit<low_bit:
                                temp = high_bit
                                high_bit = low_bit
                                low_bit = temp
                        if type(decl) == pyverilog.vparser.ast.Input or type(decl) == pyverilog.vparser.ast.Output:
                            # if type(decl) == pyverilog.vparser.ast.Output and re.match('io_pmp_\d_addr',decl.name):
                            #     decl.show()
                            ios[name] = (high_bit, low_bit)
                        else:
                            wires[name] = (high_bit, low_bit)
                # if module.name == 'RoundAnyRawFNToRecFN_1':
                #     module.show()
                #     print(nets)
                #print(type(sentence),dir(sentence))
                elif type(sentence) == pyverilog.vparser.ast.Wire:
                    name = sentence.name
                    wires[name] = (0,0)
            #print(nets)

            #exit()

            for item in module.items:
                if type(item) != pyverilog.vparser.ast.InstanceList:
                    continue
                #print(len(item.instances))
                instance = item.instances[0]
                # we extract the following parts:
                # mcell: cell name in SAED, e.g. AND2X1
                # mname: module name, e.g. ALU_DP_OP_J23_U233
                mcell = instance.module  # e.g. AND2X1
                mname = instance.name
                mcomp = mname[:mname.rfind('_')]
                ports = instance.portlist

                if mcell.startswith("SNPS_CLOCK") or mcell.startswith("PlusArgTimeout"):
                    continue
                is_target =  False
                for key_word in self.adder_keywords:
                    if key_word in mcomp:
                        cell_type = 'add'
                        is_target = True
                        break
                for key_word in self.sub_keywords:
                    if key_word in mcomp:
                        cell_type = 'sub'
                        is_target = True
                        break
                if dp_target_cells.get(mname,None) is not None:
                    cell_type = dp_target_cells[mname][0]
                    is_target = True

                if is_target:
                    print(mname)
                    # cell_name = mcell.lower()

                    #cell_type = mcell.split('_')[0]
                    index = mcell.split('_')[1]
                    # if re.match('\d+',index) is not None:
                    #     cell_type = "{}_{}".format(index,cell_type)
                    # if 'RoundAnyRawFNToRecFN' in cell_type:
                    #     print(cell_type)
                    module_info = ModuleInfo(mcell, cell_type.lower(), mname.lower())
                    for word in mcell.split('_')[:-1]:
                        if re.match('\d+$', word) is not None:
                            module_info.index = int(word)
                            break
                    # item.show()
                    # print(mcell,mname,ports)
                    for p in ports:
                        port_info = self.parse_port_hier(ios, wires, p)
                        # if some arg of the cell's port is input/output of the father module, then when the father module is instanced latter,
                        # these args should be replaced with args of corresponding port of the father module instance
                        # eg, in the following example, i1 should be replaced with w1 for cell add_x_1
                        # eg, module ALU
                        #       input [63:0] i1,
                        #       ...
                        #       CSR_inc add_x_1 (.A(i1),...)
                        #       ...
                        #     endmodule
                        #     module Rocket
                        #       ...
                        #       ALU alu (.i1(w1),...)
                        # we mantain the information of args that need to update in 'args_to_update':
                        #               {father_module_name:{(cell_type,cell_name,portname):[args need to update]} }
                        #   eg, {'ALU':{(CSR_inc,add_x_1,'A'):[i1]}}
                        # if mcell == 'RoundAnyRawFNToRecFN_DW01_inc_J71_0' and mname=='add_x_1':
                        #     print(port_info.portname,port_info.arg_list,port_info.flag_update)
                        #     print(ios)
                        if port_info.flag_update:
                            args_to_update[module.name] = args_to_update.get(module.name, {})
                            port2update = (mcell, mname.lower(), port_info.portname)
                            args_to_update[module.name][port2update] = args_to_update[module.name].get(port2update, [])
                            for arg in port_info.args_need_update:
                                args_to_update[module.name][port2update].append(arg)
                            # print(args_to_update)
                        module_info.ports[port_info.portname] = port_info.arg_list
                    # print(args_to_update)
                    target_cells[module.name] = target_cells.get(module.name, [])
                    target_cells[module.name].append(module_info)

                if target_cells.get(mcell,None) is not None:
                    if args_to_update.get(mcell, None) is not None:
                        ports2update = args_to_update[mcell]
                        father_ports_info = {}
                        for p in ports:
                            father_ports_info[p.portname] = self.parse_port_hier(ios, wires, p)

                        # print(mcell,mname,father_ports_info.keys())
                        # instance.show()
                        for (child_cell_name, child_instance_name,
                             child_portname), child_args2update in ports2update.items():
                            # find the portargs (arglist2update) of the child cell that need to update :
                            # eg, child_cell_info = (cell_type='CSR_inc',instance_name='add_x_1', ports={'A':[i1_63,i1_62...i1_0],'S':[...]})
                            #     arglist2update = child_cell_info.ports['A'] = [arg1_63,arg1_62...arg1_0]

                            for cell_info in target_cells[mcell]:
                                if cell_info.cell_name == child_cell_name and child_instance_name in cell_info.instance_name :

                                    arglist2update = cell_info.ports[child_portname]
                                    # if mcell == 'CSRFile':
                                    #     print('#############################################')
                                    #     print(child_cell_name,cell_info.cell_name)
                                    #     print(child_instance_name,cell_info.instance_name)
                                    #     print('arglist to update',arglist2update)

                                    # for every arg of args2update that needs to update, replace it with new arg
                                    for argname in child_args2update:
                                        #print("------ arg to update:",argname)

                                        replace_port_info = father_ports_info[argname]
                                        replace_arg_list = replace_port_info.arg_list
                                        new_args = []
                                        # print('replace portname',replace_port_info.portname)
                                        # print('replace arg list',replace_arg_list)
                                        if replace_port_info.flag_update:
                                            args_to_update[module.name] = args_to_update.get(module.name,{})
                                            port2update = (child_cell_name, child_instance_name, child_portname)
                                            args_to_update[module.name][port2update] = args_to_update[module.name].get(
                                                port2update, [])
                                            for arg in replace_port_info.args_need_update:
                                                args_to_update[module.name][port2update].append(arg)
                                            # print("long link")
                                            # print(args_to_update[module.name])
                                        # replace the args of the child port with new args of the corresponding father port
                                        # print(arglist2update)
                                        for arg in arglist2update:
                                            if replace_port_info.portname in arg:
                                                # print(arg)
                                                index = arg.split('_')[-1]
                                                if re.match('\d+$', index) is not None:

                                                    new_args.append(
                                                        replace_arg_list[len(replace_arg_list) - 1 - int(index)])
                                                else:
                                                    new_args.append(replace_arg_list[0])
                                            else:
                                                new_args.append(arg)
                                        cell_info.ports[child_portname] = new_args
                                        arglist2update = new_args
                                    #     print('new arglist:',cell_info.ports[child_portname])
                                    # print('#############################################')
                                    # print(cell_info.cell_type,cell_info.instance_name,cell_info.ports)

                        args_to_update[mcell] = None

                    for module_info in target_cells[mcell]:
                        module_info.instance_name = "{}_{}".format(mname,module_info.instance_name)
                        target_cells[module.name] = target_cells.get(module.name, [])
                        target_cells[module.name].append(module_info)
                    target_cells[mcell] = None
                # if we encounter a father module instance as above mentioned, eg, ALU alu (.i1(w1),...)
                #   we first parse the ports of the father module instance,
                #   then we find the corresponding relationship between args of father instance and args of target_child cell ,and replace


            #print(module.name,args_to_update)
        # for module,cells in target_cells.items():
        #     print(module)
        #     if cells is not None:
        #         for cell in cells:
        #             print(cell.cell_name,cell.instance_name,cell.ports)
        # exit()
        # print(args_to_update[self.top_module])
        target_cells = target_cells[self.top_module]
        for cell in target_cells:
            if cell.cell_type == 'sub':
                print(cell.cell_type,cell.cell_name, cell.instance_name,cell.ports)
        # exit()

        return target_cells

    def parse_nohier(self, fname,dp_target_cells,target_cells,label_region=False):
        """ parse dc generated verilog """
        #adder_cells = set()

        PIs: List[str] = []  # a list of PI nodes
        POs: List[str] = []  # a list of PO nodes
        mult_infos = {} # {mcomp:([(mult_input_wire,position)],[mult_output_wire,position])}
        nodes: List[Tuple[str, Dict[str, str]]] = [
            ("1'b0", {"type": "1'b0"}),
            ("1'b1", {"type": "1'b1"}),
        ]  # a list of (node, {"type": type})
        edges: List[
            Tuple[str, str, Dict[str, bool]]
        ] = []  # a list of (src, dst, {"is_reverted": is_reverted})
        num_wire = 0
        ast, directives = parse([fname])
        index01 = [0,0]
        adder_inputs = set()
        adder_outputs = set()
        sub_inputs1 = set()
        sub_inputs2 = set()
        sub_outputs = set()
        multdiv = set()
        muldiv_inputs1 = set()
        muldiv_inputs2 = set()
        multdiv_outputs = set()
        buff_replace = {}
        top_module = None
        # adder_in_dict = collections.defaultdict(set)
        # adder_out_dict = collections.defaultdict(set)
        # sub_in_dict = collections.defaultdict(set)
        # sub_out_dict = collections.defaultdict(set)
        positions = {}
        pi_positions = {}
        for module in ast.description.definitions:
            if module.name == self.top_module:
                top_module = module
                break
        assert top_module is not None, "top module {} not found".format(self.top_module)
        print(len(top_module.items))
        for item in top_module.items:
            if type(item) != pyverilog.vparser.ast.InstanceList:
                continue
            instance = item.instances[0]

            # we extract the following parts:
            # mcell: cell name in SAED, e.g. AND2X1
            # mtype: cell type with input shape, e.g. AND2
            # mfunc: cell function, e.g. AND
            # mname: module name, e.g. ALU_DP_OP_J23_U233
            # mcomp: module component, e.g. ALU_DP_OP_J23
            mcell = instance.module  # e.g. AND2X1
            mname = instance.name
            ports = instance.portlist
            mtype = mcell[0 : mcell.rfind("X")]  # e.g. AND2
            mfunc = mtype  # e.g. AND

            # pos = re.search("\d", mtype)
            # if pos:
            #     mfunc = mtype[: pos.start()]
            mcomp = mname[: mname.rfind("_")]
            if mcell.startswith("SNPS_CLOCK") or mcell.startswith("PlusArgTimeout"):
                continue
            fanins: List[PortInfo] = []
            fanouts: List[PortInfo] = []
            # if 'add_x' in mcomp or 'alu_DP_OP' in mcomp:
                #print("\n",mcell,mname)
                #adder_cells.add(mcell)
           # exit()
            dp_inputs,dp_outputs = [],[]

            # adder_cells: { ([],[],(None,[]))}
            for dp_cell in dp_target_cells.keys():
                if dp_cell in mcomp:
                    dp_inputs = dp_target_cells[dp_cell][1]
                    dp_outputs = dp_target_cells[dp_cell][2]
                    break

            for p in ports:
                port_info = self.parse_port(mcomp, target_cells,p,index01,dp_inputs,dp_outputs)
                if port_info.ptype == "fanin":
                    fanins.append(port_info)
                elif port_info.ptype == "fanout":
                    fanouts.append(port_info)
                # else:
                #     assert port_info.ptype == "CLK"

                if port_info.is_adder_input:
                    #print(port_info.)
                    adder_inputs.add(port_info.argname)
                    #adder_in_dict[port_info.input_comp].add(port_info.argname)
                if port_info.is_adder_output:
                    adder_outputs.add(port_info.argname)
                    #adder_out_dict[port_info.output_comp].add(port_info.argname)
                if port_info.is_muldiv_output:
                    multdiv_outputs.add(port_info.argname)
                if port_info.is_muldiv_input1:
                    muldiv_inputs1.add(port_info.argname)
                if port_info.is_muldiv_input2:
                    muldiv_inputs2.add(port_info.argname)
                if port_info.is_sub_input1:
                    #print(port_info.)
                    sub_inputs1.add(port_info.argname)
                    #sub_in_dict[port_info.input_comp].add(port_info.argname)
                if port_info.is_sub_input2:
                    #print(port_info.)
                    sub_inputs2.add(port_info.argname)
                    #sub_in_dict[port_info.input_comp].add(port_info.argname)
                if port_info.is_sub_output:
                    #print(port_info.argname,port_info.argcomp,port_info.portname,port_info.position)
                    sub_outputs.add(port_info.argname)
                    #sub_out_dict[port_info.output_comp].add(port_info.argname)

                if port_info.flag_mult:
                    multdiv.add(port_info.argname)
                if positions.get(port_info.argname,None) is None:
                    positions[port_info.argname] = port_info.position
            if not fanouts:
                item.show()
                print("***** warning, the above gate has no fanout recognized! *****")
                # do not assert, because some gates indeed have no fanout...
                # assert False, "no fanout recognized"
            inputs = {}
            #print(mfunc,mname)
            for fo in fanouts:
                # if fo.flag_mult:
                #     mult_infos[mcomp].fanouts[fo.position[0]] =  mult_infos[mcomp].fanouts.get(fo.position[0],set())
                #     mult_infos[mcomp].fanouts[fo.position[0]].add((fo.argname,fo.position[1]))
                if mfunc == "HADD":
                    if fo.portname == "SO":
                        ntype = self.hadd_name_dict["hadd_s"]
                    elif fo.portname == "C1":
                        ntype = self.hadd_name_dict["hadd_c"]
                    else:
                        print(fo.portname)
                        assert False
                elif mfunc == "FADD":
                    if fo.portname == "S":
                        ntype = self.hadd_name_dict["fadd_s"]
                    elif fo.portname == "CO":
                        ntype = self.hadd_name_dict["fadd_c"]
                    else:
                        print(fo.portname)
                        assert False
                else:
                    ntype = mfunc

                if 'AO' in ntype or 'OA' in ntype:

                    num_inputs = ntype[re.search('\d',ntype).start():]
                    ntype1 = 'AND' if 'AO' in ntype else 'OR'
                    ntype2 = 'OR' if 'AO' in ntype else 'AND'
                    if 'I' in ntype:
                        output_name = '{}_i'.format(fo.argname)
                        nodes.append((output_name,{"type":ntype2}))
                        nodes.append((fo.argname, {"type": 'INV'}))
                        inputs[fo.argname] = [output_name]

                        # edges.append((output_name,fo.argname,
                        #               {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                    else:
                        output_name = fo.argname
                        nodes.append((output_name,{"type":ntype2}))
                    inputs[output_name] = inputs.get(output_name,[])
                    for i,num_input in enumerate(num_inputs):
                        if num_input == '2':
                            h_node_name = '{}_h{}'.format(fo.argname,i)
                            nodes.append( (h_node_name,{"type":ntype1}) )
                            inputs[h_node_name] = inputs.get(h_node_name,[])
                            inputs[h_node_name].append(fanins[2*i].argname)
                            inputs[h_node_name].append(fanins[2*i+1].argname)

                            inputs[output_name].append(h_node_name)
                            # edges.append((fanins[2*i].argname,h_node_name,
                            #              {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                            # edges.append((fanins[2 * i+1].argname, h_node_name,
                            #              {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                            # edges.append((h_node_name,output_name,
                            #               {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                        elif num_input =='1':
                            inputs[output_name].append(fanins[2*i].argname)
                            # edges.append((fanins[2*i].argname,output_name,
                            #              {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                        else:
                            print(ntype,i,num_input)
                            assert  False
                # elif 'NOR' in ntype or 'XNOR' in ntype or 'NAND' in ntype or 'IBUFF' in ntype:
                #     ntype1= None
                #     if 'NOR' in ntype:
                #         ntype1 = 'OR'
                #     elif 'XNOR' in ntype:
                #         ntype1 = 'XOR'
                #     elif 'NAND' in ntype:
                #         ntype1 = 'AND'
                #     elif 'IBUFF' in ntype:
                #         ntype1 = 'NBUFF'
                #     h_node_name ="{}_h".format(fo.argname)
                #     nodes.append((h_node_name,{"type":ntype1}))
                #     nodes.append((fo.argname,{"type":"INV"}))
                #     inputs[fo.argname] = [h_node_name]
                #     inputs[h_node_name] = inputs.get(h_node_name,[])
                #     for fi in fanins:
                #         inputs[h_node_name].append(fi.argname)
                #     # edges.append((h_node_name,fo.argname,
                #     #               {"is_reverted": False, "is_sequencial": "DFF" in mtype}))
                # elif 'DFF' in ntype and port_info.portname=='QN':
                #     #print(ntype,port_info.argname,port_info.position)
                #     ntype1 = 'DFFN'
                #     h_node_name = "{}_h".format(fo.argname)
                #     nodes.append((h_node_name, {"type": ntype1}))
                #     nodes.append((fo.argname, {"type": "INV"}))
                #     inputs[fo.argname] = [h_node_name]
                #     inputs[h_node_name] = inputs.get(h_node_name, [])
                #     for fi in fanins:
                #         inputs[h_node_name].append(fi.argname)
                #     # edges.append((h_node_name,fo.argname,
                #     #               {"is_reverted": False, "is_sequencial": "DFF" in mtype}))

                else:

                    pos = re.search("\d", mtype)
                    if pos:
                        ntype = ntype[: pos.start()]
                    # if 'DFF' in ntype :
                    #     ntype = 'DFF' if port_info.portname =='Q' else 'DFFN'

                    inputs[fo.argname] = inputs.get(fo.argname,[])
                    for fi in fanins:
                        # dff ignore SET/RESET/CLOCK
                        # if 'DFF' in ntype and fi.portname!='D':
                        #     continue
                        #if ntype == 'NBUFF' or ('DFF' in ntype and fo.portname=='Q'):

                        if ntype == 'NBUFF':
                            buff_replace[fo.argname] = fi.argname
                        else:
                            inputs[fo.argname].append(fi.argname)

                    #if ntype == 'IBUFF' or ('DFF' in ntype and fo.portname=='QN'):
                    if ntype == 'IBUFF':
                        ntype = 'INV'

                    if buff_replace.get(fo.argname,None) is None:
                        nodes.append((fo.argname, {"type": ntype}))
            #print(len(inputs))
            #print(inputs)
            # if len(mult_inputs)!=0 :
            #     for fi in fanins:
            #         if fi.position is not None and fi.position[0] in mult_inputs:
            #             mult_infos[mcomp].fanins[fi.position[0]] = mult_infos[mcomp].fanins.get(fi.position[0], set())
            #             mult_infos[mcomp].fanins[fi.position[0]].add((fi.argname, fi.position[1]))

            for output,input in inputs.items():

                for fi in input:
                    edges.append(
                        (
                            fi,
                            output,
                            {"is_reverted": False, "is_sequencial": "DFF" in mtype},
                        )
                    )


            # for fi in inputs:
            #     for fo in fanouts:
            #         edges.append(
            #             (
            #                 fi.argname,
            #                 fo.argname,
            #                 {"is_reverted": False, "is_sequencial": "DFF" in mtype},
            #             )
            #         )

        print(index01)
        #print('mult info')

        # for mcell,info in mult_infos.items():
        #     #print('----{}'.format(mcell))
        #     fanins = info.fanins
        #     for port in fanins.keys():
        #         new_args = sorted(fanins[port], key=lambda x: x[1])
        #         temp = []
        #         for item in new_args:
        #             temp.append(item[0])
        #         fanins[port] = temp
        #     fanouts = info.fanouts
        #     for port in fanouts.keys():
        #         new_args = sorted(fanouts[port], key=lambda x: x[1])
        #         temp = []
        #         for item in new_args:
        #             temp.append(item[0])
        #         fanouts[port] = temp
            #print(fanins)
            #temp = sorted(fanins.items(), key=lambda x: x[1])
            #print('fanins:',info.fanins)
            #print('fanouts:',info.fanouts)
        print('num of nbuff/dff_q:',len(buff_replace))
        new_edges = []
        for edge in edges:
            if buff_replace.get(edge[0],None) is not None:
                new_edges.append((buff_replace[edge[0]],edge[1],edge[2]) )
            else:
                new_edges.append(edge)
        edges = new_edges
        print(
            "#inputs:{}, #outputs:{}".format(len(adder_inputs), len(adder_outputs)),
            flush=True,
        )
        #self.label_mult(nodes,edges,mult_infos)
        #print(adder_inputs)
        gate_names = set([n[0] for n in nodes])
        pis = []
        for (src, _, _) in edges:
            if src not in gate_names and src not in pis:
                nodes.append((src, {"type": "PI"}))
                pis.append(src)
                # if "1'b0" in src:
                #     print("1'b0")
                # if "1'b1" in src:
                #     print("1'b1")
            # if "1'b0" in src :
            #     nodes.append((src,{"type":"1'b0"}))
            # if "1'b1" in src :
            #     nodes.append((src,{"type":"1'b1"}))
        #print(pis)
        if label_region:
            g = nx.DiGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(edges)
            rg = g.reverse()
            internal = set()

            for m in adder_in_dict:
                in_nodes = list(adder_in_dict[m])
                out_nodes = list(adder_out_dict[m])
                forward_reachable = set()
                backward_reachable = set()
                for i in in_nodes:
                    fw = dict(nx.bfs_successors(g, i, 6))
                    for t in fw.values():
                        forward_reachable.update(set(t))
                for o in out_nodes:
                    bw = dict(nx.bfs_successors(rg, o, 6))
                    for t in bw.values():
                        backward_reachable.update(set(t))
                internal.update(forward_reachable.intersection(backward_reachable))
                i_not_r = 0
                o_not_r = 0
                for i in in_nodes:
                    if i not in backward_reachable:
                        print(i)
                        i_not_r += 1
                for o in out_nodes:
                    if o not in forward_reachable:
                        print(o)
                        o_not_r += 1
                # print("{}: iNOT={}, oNOT={}".format(m, i_not_r, o_not_r))
            for n in nodes:
                n[1]["is_adder"] = n[0] in internal

        else:
            count = 0
            for n in nodes:
                n[1]["is_adder_input"] = n[0] in adder_inputs
                n[1]["is_adder_output"] = n[0] in adder_outputs
                n[1]["position"] = positions.get(n[0],None)
                if n[0] in multdiv:
                    n[1]['is_adder_input'] = -1
                    n[1]['is_adder_output'] = -1
                n[1]['is_mul_output'] = n[0] in multdiv_outputs
                if n[0] in muldiv_inputs1:
                    n[1]['is_mul_input'] = 1
                elif n[0] in muldiv_inputs2:
                    n[1]['is_mul_input'] = 2
                else:
                    n[1]['is_mul_input'] = 0

                n[1]['is_sub_output'] = n[0] in sub_outputs
                if n[0] in sub_inputs1:
                    n[1]['is_sub_input'] = 1
                elif n[0] in sub_inputs2:
                    n[1]['is_sub_input'] = 2
                else:
                    n[1]['is_sub_input'] = 0

        print('num adder inputs:', len(adder_inputs))
        print('num adder outputs:', len(adder_outputs))

        print('num muldiv inputs1:', len(muldiv_inputs1))
        print('num muldiv inputs2:', len(muldiv_inputs2))
        print('num muldiv outputs:', len(multdiv_outputs))


        print('num sub inputs1:', len(sub_inputs1))
        print('num sub inputs2:', len(sub_inputs2))
        print('num sub outputs:', len(sub_outputs))
        #print(adder_cells)
        #print(nodes)
        return nodes, edges

    def parse(self,vfile_pair,hier_report):
        hier_vf, vf = vfile_pair[0], vfile_pair[1]
        # if 'hybrid' not in vf:
        #     continue
        # parser = DcParser("BoomCore", ["alu_DP_OP", "add_x"])
        dp_target_cells = self.parse_report(hier_report)
        # exit()
        target_cells = self.parse_hier(hier_vf, dp_target_cells)

        nodes, edges = self.parse_nohier(vf, dp_target_cells=dp_target_cells,target_cells=target_cells, label_region=False)
        return nodes,edges

    def label_mult(self,nodes,edges,mult_infos):
        g = nx.DiGraph()
        #print(nodes)
        temp_nodes = []
        for nd in nodes:
            temp_nodes.append(nd[0])
        temp_edges = []
        for edge in edges:
            temp_edges.append((edge[0],edge[1]))
        g.add_nodes_from(temp_nodes)
        g.add_edges_from(temp_edges)
        rg = g.reverse()
        internal = set()

        for mcell in mult_infos.keys():
            fanins = mult_infos[mcell].fanins
            fanouts = mult_infos[mcell].fanouts
            fanin_args,fanout_args = [],[]
            for args in fanins.values():
                fanin_args.append(args)
            for args in fanouts.values():
                fanout_args.append(args)

            in_nodes = None
            out_nodes = None
            forward_reachable = set()
            backward_reachable = set()
            for i,fanout in enumerate(fanout_args[0]):
                intersect = None
                for j in range(i):
                    if j < len(fanin_args[1]):
                        all_paths = nx.all_simple_paths(g,fanin_args[1][j],fanout,cutoff=15)
                        all_paths = list(all_paths)
                        #path = nx.shortest_path(g,fanin_args[0][j],fanout)[:-1

                        path_union = set()
                        for path in all_paths:
                            path_union = path_union | set(path)
                        intersect = path_union if intersect is None else path_union & intersect
                        #print('src:{},dst:{}'.format(fanin_args[0][j],fanout))
                        #print(list(all_path))
                for k in range(i):
                    if k < len(fanin_args[0]):
                        all_paths = nx.all_simple_paths(g, fanin_args[0][k], fanout, cutoff=15)
                        all_paths = list(all_paths)
                        # path = nx.shortest_path(g,fanin_args[0][j],fanout)[:-1
                        for path in all_paths:
                            if i==3 and 'div_DP_OP_279J21_124_314_n1132' in path:
                                print(path)

                        path_union = set()
                        for path in all_paths:
                            path_union = path_union | set(path)
                        intersect = path_union if intersect is None else path_union & intersect
                print(i,fanout,intersect)
            for i in in_nodes:
                fw = dict(nx.bfs_successors(g, i, 6))
                for t in fw.values():
                    forward_reachable.update(set(t))
            for o in out_nodes:
                bw = dict(nx.bfs_successors(rg, o, 6))
                for t in bw.values():
                    backward_reachable.update(set(t))
            internal.update(forward_reachable.intersection(backward_reachable))
            i_not_r = 0
            o_not_r = 0
            for i in in_nodes:
                if i not in backward_reachable:
                    print(i)
                    i_not_r += 1
            for o in out_nodes:
                if o not in forward_reachable:
                    print(o)
                    o_not_r += 1
            # print("{}: iNOT={}, oNOT={}".format(m, i_not_r, o_not_r))
        for n in nodes:
            n[1]["is_adder"] = n[0] in internal
def main():
    report_folder = "./report"
    folder = "./implementation/"
    # folder = "../dc/boom/implementation/"
    total_nodes = 0
    total_edges = 0
    ntype = set()
    vfile_pairs = {}
    for v in os.listdir(folder):
        if not v.endswith('v') or '10' in v or 'auto' in v:
            continue
        if v.startswith('hier'):
            vname = v[5:-2]
            vfile_pairs[vname] = vfile_pairs.get(vname,[])
            vfile_pairs[vname].insert(0,v)
        else:
            vname = v[:-2]
            vfile_pairs[vname] = vfile_pairs.get(vname, [])
            vfile_pairs[vname].append(v)
    vfile_pairs = vfile_pairs.values()
    print(vfile_pairs)
    for vfile_pair in vfile_pairs:

        hier_vf,vf = vfile_pair[0],vfile_pair[1]
        hier_report = os.path.join(report_folder,hier_vf[:-1]+'rpt')

        # if 'hybrid' not in vf:
        #     continue
        hier_vf = os.path.join(folder, hier_vf)
        vf = os.path.join(folder, vf)
        print("parsing {}...".format(hier_vf))
        # parser = DcParser("BoomCore", ["alu_DP_OP", "add_x"])
        parser = DcParser("Rocket", adder_keywords=['add_x','alu_DP_OP','div_DP_OP'],sub_keywords=['sub_x'],hadd_type="xor")
        adder_cells,sub_cells = parser.parse_report(hier_report)
        #exit()
        target_cells = parser.parse_hier(hier_vf,adder_cells,sub_cells)

        nodes, edges = parser.parse_nohier(vf, adder_cells=adder_cells,sub_cells=sub_cells,target_cells=target_cells,label_region=False)
        print("nodes {}, edges {}".format(len(nodes), len(edges)))
        for n in nodes:
            ntype.add(n[1]["type"])
        total_nodes += len(nodes)
        total_edges += len(edges)
        #break
        print(ntype)
        break
    print(total_nodes, total_edges)


if __name__ == "__main__":
    # dc_parser("../dc/simple_alu/implementation/alu_d0.20_r2_bounded_fanout_adder.v")
    main()
    # cProfile.run("main()")
