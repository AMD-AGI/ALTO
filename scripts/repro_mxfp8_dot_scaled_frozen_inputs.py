#!/usr/bin/env python3

import base64
import os
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELOPTIMIZER_DIR = REPO_ROOT / "modeloptimizer"
KERNELS_DIR = MODELOPTIMIZER_DIR / "kernels"
MXFP8_DIR = KERNELS_DIR / "mxfp8"
TESTS_DIR = MXFP8_DIR / "tests"

SHAPE = (64, 64, 64)
MXFP_FORMAT = "e4m3"
OUTPUT_DTYPE = torch.float32
FP8_DTYPE = torch.float8_e4m3fn

# Frozen input tensors taken from the stable failing case used by
# `scripts/repro_mxfp8_dot_scaled_vs_emulated.py`.
A_LP_B64 = "9Wno7/He8nHr6V3wbN9hVO31eWZ40uDeZN3fX/Xh8uLgdORYaXPudfDiceXubXzfbHL4c/FTdeDd8vtmdPFr71XQSlBMflbYyc3NUt9SSVVTy1HQvMpZVcrLV9bX0lLTcXnK6HTf9mRd8m7s7Odv8FLWdMZi3OFq5Wpqy23ze9vic1js4+Fr33rp7PLs+nNxWGFaa3HT5fF29lZwafrnSnF57tx5eP3ndtn3cnp78/q9bnfy+HD+ePhzdunjYHD9cvNr48348+do411Xz17x33heY9v3+uHsY3dhaHTw+Gjl4fVLUvRq2F3r+lV1cvfob3NZcurf4HJseXBzb/Zu9klFysA4UkJQVFLC2UnVScnT0c7Ny0zQR8BPycV7VafCcu7l4WXt2nLh6Wvw9vrmce358WvhWVNi2WF67ndnR+R0+/E88sbMaXj2cHVz7VD1avbU6vtx8GXWcXHg53R2avDf8W/zblTyc2Xx62x1cm/heW9YS21adHFn9Onj5OxhZdPi4uJxeGzyeGtW5Ej0cfHpb1t88GrdZ99p3nJuamXobOJicuxwce3zXmnr6vFieWvqee3b52zu8HHnZubyZmdj3NP7WOBr42vpMeTQxWTSW+DY5FvmYVtf3evU1jzTW0bBVlQZ0chQ0smwykIZWjVWzTJD1UTNxXlHwjnJq8vu42/ldG/TcGbwNOhz5OlxbW56cd5x2WJ1bfFp+G3xd9LSUUxR1eXQzl3T0dq8WMbgeuDRyuDYT0/Mu83MIFPRZ/HxZuzYXDTu7+Xt5fvz6k1r9ezm5O3Z7Wl76vRxcnZh2Gtra/D37/Fn73PpeNHuauP13V7t6OvvbnTvcmn4dVBgNP43Sr/a5Fg6VFNSXFhfSlPOUF1ayFvUQNbYz+BR4dJNYOhC4tHBz8nlOUBh2t3jaMLmYuH7P1G64FnVW2jy9/NnafhsbWr6XU12eP3p+v163153+3dw8vFvcWD2+Tpw9G+3ZuVubNl59On3ZWJ18mZ3NGzn+3nr8+10Z3dq3/Vx33fxbHJ16fXJ8eZ1+OzUWu3S2Xpy725tX9jvG/hnTtXh1Wpi62Nq5fFw5GpX3e1gefXeYPH4c9niYmlsXnLE1tfOZutwXXDZ6vBu+e5y4fDdcmRg8up2duEo6GrjX2XXz1XaU+NMV2BhXDth+2RH5l5h3mDVaVFqSN1kVeJacHDq8vR2dvNcdF/7/Uv4fu50/nXwa/778Hjte3Fo+exK3VF43O5qafpj8tls9NnqY+9p4UnpYkvpbO3sa+9c5N5xVfL4aPTj7u7X4dVg9WXpbfZ1cVBQ6WhzcVTwdu9q7XTlb/Nx89vicOt57nFp6OFyfVVnXnHqeO5pZNhoeXFg4Xpvxu7iZPTxam7y611p8+Nm8mjxe29h3CFrT/Jh7PXj+HZsa2lw6mlxVW5hdOPidOtv8uNy8+r32URzc9f1dPh4Zvn5affrcvdq/XLrWvt6fXpeWfjq6Vzgdutz7/L7+fBp7vD3bO1z6WptYGx5YtlLbeBE8fPwbfFZZWNo9GXqYu9r8VxUa+zv7+5ka+ns4ePfYGRt9fh2bnZvevEZPT3Lqnq4ukU7Mza/NR82Lh6nQ0BDFq/EQbnHvDxAOFTbaudf4mZvdulsa/brbbly4/Fn5W9sdXVW7mX19HrD/mJbYHj532xw9VH3bOr0sOjyZnDu8+Hp8fVZaVhr8e7LwEtCzczHPLLCSf6wzT3LwTPKTcLGS8nMTc09TzlGyup68O/lzuVm43blc1huamrqdW3FdnVn+PXuXWzfY3LfcllpUfBu3fB08PZn8+Xt1/jHc+Dk7uDy79VX+FxkZ+jy/HPy831q6N97cWjNevthaFnxb2zt/rVm9nr85Hx4cHp5ZHLU3nlwa3h1ZfJ78HH85XzeePJ+TVH+8fnf9F5t+mRy/GZs/vrwbv32eG99+/1qaPbld/pvbXMx9Pvz/nFv4/CZ5OPv3PnTZ+76dtnh6uXj99Nrcmzs6+FxY3Jp+eNt93dr4OdTZF7qbWLncd1sa/BZY2VbZEx0YN549GnxcPDi6mNJ8nln7fnsZ+/ZdXtq8Grwd3bc8ejk8XJwbGB08PFV8l/h5WRw9kXw8u70cnZxVfJt60Z7UeF0eMT86XttfWBkYXh6b3p32frx8u7DeFtxamn062x09vJ6afLb7GtvcnHubexw7PNWw3J4dGnDZNtx+Nns42Jr0mrxedJeeD7hZNntcvFyU/Ft6ur15+1xbXDw73tr81zMcvJ6wfbncWpqaH7u7Xvmb1p4bPP3b+7l7mtcajXA02vieuzzZ3bxdGnqbW11bG5s8eZ6ynDUcORg4/ZV3V1j3l3q9kdGvq3EMTo6NTq0Lrc4vsKxSL9BfTCgvSo3a0EXtqe0M+Lg7fXxdWzt8Gfxb9xy8ezx8MxuZfXVcMTs+e7U73nq4el0Z/VkfWzW/Ox5ceR0dXr37N3N8W1ndPNf4WbwU+9u9vDRXuVh6PTlbGrydcxZZWpn5GTjRt33+/Dv7PLydunzWe1T9Wrg8flM7W5611/rYOnwcm9M43TtdG30anHp12RoaWJb7HXs8n7vd+B55Pz6XVhz2XL0eGzg6nZzfW/mbXhWct3s7Px18O7Y3eRt9nXsZtH3cnn1bXrtbvHhcvJx+l1n7PJ7enVr2/Zz6PXy/fLo/GlX9XJ5fPztc/Nn3OHocd/0aWbq59T0aW3Ucuj4U/Ln8drB7m1Y8fT3c3Tc7uVpanLnaF5ZWvPtcurkcmly4fJd4HDW8+h432xtbG3qbEBXZ/hu8djrrXRoaN3IbfJwaPTuavbvdFlRau7z+LxUMchU0sNEy09T08k71j8/2UHC0NvJ1FHEx0e7WdRw8/Tj+fHucfnl7WjyUvlyZuJ3zWtwaWfnYPVu7mE9ZdNfb3HxcfVn88BEb/Pn6mL04upl2XFsb+1n8Nzv+m7uwTksMESxQjPDqzLHw8LEN5O2/jbBQ5WuwUUlvCI3NR/S+F9ecPHq1Ox4bOvwemr24llvZunk023y7WVbX1hyZ2TsZnls9+lsyOnwaVv0aeLy5VrpaWVp8njm8Wr49HFhcvhk4URdXVlcb+nZTHTwZHT09G1x9fL06PDmbMVvce5iVthlZODt6trt6dlb8Oz42rni5dpn1uFr4/NsVujfS/pgV3VwbN3dYF15dui+fW128v7Dfm1s9nhP8HBueGl28O/qdnJkYtlybOncZNb2c9pm7dlteGhh3WhVafJ08nnzd2lhcGRdcmx51e1vZffSbWpp+PTuenVud3JyYPFicGh17+Vi4O5o6WtD63Tp71th8XXoylrw+PL2T2Rv7vDtevJz9fr08mFdZ1xgdlr36PTx4nTAcPz5cH5pUHB6cHZKzatBwTkaNc2/RExGPlCeSTFBjT1AQ8q7Nkz+wcXNSXbb8fb6aeho8fbwcHjqY+xt72FzT3ftcWZw7XNewHbw+cz3aPLqcGVx8GTv62NYW/F29HBg8WtxY+9lcel01Gxaa3hnasxw6vFY6W/mdnX4QGZqcmX3ePHpbL3nZvXrauLpbNrdedNI8W7k6PJr5uVsdOnkcWBsR3R4cWxR4G5rZPPxyOg+4ufr4/bjamb2cnFkbGJjb3brcGhw4u1f+Gpz/vt671t6cWL78fF5d3vdaWt0YvJsYnP98Xt1/v7pcfVSUvn4c3nlYPh3XW7g8vBH49Xt6mlx8etyenRtenJvVcnVUFHOOy1D2sLUQtLCTrZ5WEjSxsnE0VBQVry42EhSX/D7Xu352e/42nVC88x3den1d+1n7XZpZPbx7Xpyd2framby6nlu83JzaOvv8G338fNn+2FuZ+TraFpdR+I773J18+Vk6fDrd+xq0uZxfklictrn68jwZXDf+uvtaOVd+Xnr7+3k4XNr7drs+Fz86Oz2anLx7+xraMjbdetzZ/fvb9LrafJpcmlh43HoaG552fDt9e1t6fTk6XRpVOlhdHNp/W9f1nlm8+36eHHz5Obz6fJh/OJ243F12HL47+66x0VQz85O0k/MzsU/0U9SwUbKw7L7wVPIQ0lVUbhPSfz2d3D5/vly6fp0dGJ77Hh5KnT0an3Zfvl9clr59WfzQs5U2FnUecLXSNC/Rz/L2rQxO8A6O7tTw9UzRVRByZZ3c3V47dze4+zd+vLq6WLq4/Ru9GThU3FpuGZL72dVexvDMklQ/tdA1UvMUFDTREoxz0tdxces1cs219fQ0crU6Gz2cmNbderb6N3u9Gp0/Ov2UvBWaXN3b2Dq6fbtaWrpWe5p5ntf82/l6HHp+PNybuBvYXBlZm5h2uHUa8Ffamn03Wxb0ez363FY6/hdSnJu8fD2Um91dHPU8G3272xbc3V+bvr48Xrq2fHjfHX18eLz9/h3+H379XJsdP748On3Xt758nlyUn7ys/n4cPz3+HVvbPhr9vt973x9zftjb7bV+NFYSlva2FdTVWNSysvLUWJUXUtRTcrhWlZJyUhb3+dsMm/k7Wvq6/Di8+nyVnNv5PJy8fBceFJL8Wdh4u/y7HnuzFfxdmpWYXNZ7OzhafnsY2/MYnL3ckTkY9rJbrq6SEmqQkq9vbtPwUNNQL0hx0lHxs79Okq4x85ERLy/ceFu3O91Zvby4vDcX+1ta1bid2rh2PHxcvlv4WxvXnD2XvZt5GzyXvDm8uNm+9n372BO8+fidOK9WWrs8elx8Olka3Hldlbw6W14Ym7w4Hh4+Wnp60FpZGlK6Wz1VW5X629ybPDtXebjd2ns8uhde2to5+1xaPF0YHfsce9r1/Rt5mTpZ2HnbnDx7fppXdFe8urh4dpnZHQhcG/uVfT38fTxa3la5uPrcux4VfJw7vhqY3PmZ3Bnb2hf8XNpZuNo+Ex5cPLgc/rn9Gn1cmP6c+1k2GDy4W75cXbsznHf7OP4aWjm2fBoad3n7vhw8u31cWRd6PZs6nric3TUbGP4ctx08WnqQWX4aXnteGxx1fNZ3UFrdFfn73LnYWl05OlM4+3vd2pZ5G9v6WnSaeboaOzk9znsdN5xZ3vod9vycWXccfNxWWzYXnFqW3Xu+mnjbGNda3Al6nLt5/JtbOHpWlXxaXNycnZlbOdu8nRdWGJkcOpgW87pVuvR/OlxX1Lj52rv7HHs5fBu2XFkbEN49vH2YMh3YmhiwurvcGtw63JLaPZs8ubu5u9qc2ng+dtqcvTnZ/RyaPNqZ+td0Or0QF7o79zna+bfcnH1WuNy3XLz82FscWpvaV5xaHbw8njpwcUhNL62QEZHSFGKJML6R0gwTcixPTW6qb7QQ1LFTUp8fGr+bubqdGx5eXN+9Vd1dGpS9Xlu+vxH9Wv342zq/Tvqw3xpb23lU+3Z5nlSbvzrzXDzPDziddDpbtXq8Ov6R9PBSUg0UE5FyE3JsUo/xatGxb/EOLPMSlBLtsH9q8XgY2HTae1waWDg8OnuW3fbfHVvbvnq32t0WlHlWNpv0w=="
B_LP_B64 = "/Wnw1+HG8nHr6V3IbN9hVO39eW540uDebN3fX9Xh8uLgdORYaUPuffDiceXubXzPbFr4Q/lTfdDd0vtudPFr73XoalBUfm7w4eXlQvdqYW1r62nw1OJxberjb+7P6mrrcXnK6HSv9mxd8m7s7Odv4FK+dJZq3Ola5Upq023ze9vqc2DU08lr33rp7Mrs+nNxWGlac3HT5fF+9lZwSfrnSmlx5tRxQPXnbtHvanJz6+K1Tm+6+Gj+YPBLbunbWGj1evNzy73g8+do410vz17x33hmY+P3+uHsa3dhaFTw+Gjl4fVLUsRq4F3r+lV1cvfYb1tZQvLf6GJsWXB7b/Zu9nFl8shIWmJwdHLi0Wn1aenz+e7162zwZ+hv6eV7dcficu7l4WW92nrh6Wvw9vrmYe3h8TvpWVtS2UF69ndnR+R8+/kk4q7MaXj2cE1z7VD1av7U8vtx8GXecXHgx3R2avDf8W/zPlT6c2Xx62x1cl/hYW8oU21iZHFH9PHj5OxhbdPqytJZeGzyeGsu5Ej0cfHxb2N88Grdb99p3lJuamXobOJicrxwee3zXmnr6vFSeVPqSfXb71zu0HHvZubyZn5z9Mv7UPB783v5GfTg1XTic/Dw9Gv2cXNv7fvE5kzje2bhdnQJ8fBw8unQ6mI5alVe7SJr9Wzd5Xln6lnpy+v243fNZFfTcGbwNMBz5OlxbXZ6ed5x2WJ9bfFp2G3xd+rqaWRpvf3w5nXr6fLUcM74evi56vj4V2fE0+3kOGvpb/H5TtzAXDTu7+XF5fvz6k1z9fTm5O3Z9Wl76tRxcnZh2Gtra8D39/Fn73PpeNHeasv1rWbt8NvvTnT3cmn4dXB4VP4/Stfy/HBSRGtqdHB3amvuaHVy4HvsWO7Q5/hp8eJdcPgi8unR39n1SVBx2u3beKL+cvn7T0HK+Gnla3jy7/NHUdhkZWLyVR1ucPXh8v1y31Zv829w6ulnSVju8Tpw9G+3NuV2bNl59On3ZVJ12mZHPGzv63nL8/V0Z3dq5/V5x2fZbHJ16fWh8eZ1+OzcWvXS2Xpy925tX7jvG/hnTtXh1Tpi82Nq5fFw5GpH3dVgSf3eaOH4U9nqYmlsXnrE3r++TutwXXDZwvBu+e5y6fDlcmRg8vJ2duEI6Grjb3Xn32W6Y/tcZ3BxbEtx+3Q/9j557njVeUF6YO10ZfJaaHDK2tRubutUbC/z9UPwd+5s/m3oY/f76HDlU2lg8exK3VF4rO5yafpj8tls9MnqS+856UnxUkvJbPXsa+9c7N55PeLgaPTj7u6v4dVg9WXxbf51cVBQ8WhzcTTwdu9q7XTlb8Nx+9vicOt57nFZ6MlyTV1nZmHqWO5xZNhoeXlg6WJfru7iZPTxQm7y611p++Nu8mjxe3dh3CFLT/Jh7PXj+HY8a3Fw6mlxVW5hZOPKdLt38uti88r34URzc9f1bPhYTtnxYe/jasdi9WrjUvtyfXJWUfDq4VTYTuNr5/L7+fBpvvD+bO1z6WptYFx5StkbdeBM4fPQbflZZWNo/GXySt9T8VxUa+zH7+5ka+n04evfYGRt/fh2blZvevFJbW372nro8nVrY2bvZU9WXjbXQ3hzTs/0Uen+7GxwaFzbcs9PymZvdulsQ/brbbly6/Fv5W9sdX1W7mXV9HrD/mJbYHjJ33Rw9VH3bOr0oOjaZkD28+nZ8dVZcVhr8e776HtS5dzvZNrqcf7Y9WXz6WPyferuc/H8dfVlV2Fu8up68O/lnuVu43blc1hualrqXW2VfnVv6PXOXXTfY3LfellxOeBW3fB08PY/8+Xt1/jPc+jk7uDy99VX+DxkZ+jq9Gvq60Vi6NdzaWDFcvNJYDnpN2zl/p1eznL83HRwaHpxZFK8vnFoY3BtNepz6Gn05XTecOp2RVH36fG37FZl8lxq9F409vroZvXucGd14/VKYL7lb/pXZUsp9PPr92l34/iF1Mvv3PnTZ8b6dtnh6u3j/tNrcmz06+FxQ3Jp+eNt93drsOdbZF7qbWLncc1sU/Apa2VjVExUYOZ49GnxePDq0lMx8nln7fnEZ+/ZdXty8HLwd3bc+ejk8VJwbGB08PFV8i/h7WRw9kXw8u7kcl5xJfpt8zZ7MeF8eMT86XtlfUBMQXByZ3JvqfLp6ua7eFNxYmHs42xs7upSYerT7GtvcnG+bfRw7PNWw3J4ZGmrZKt5+OHc40Jr2mrxedJmeEbJVMHtcvFyU8lt6ur15/VxdXDw73tz81zMUvJ6wfbncWpqOH727Xvmb1p4bOP3V+619mtkWjWg03Pieuzzb3b5XFnSbW11bG5E8eZ6ynDccOxg4/ZV5V1j3j3q9kd27t30YTpqbWrkXudo7vLReNdxfWjQ9Upne3FP5tfkY+rg9d3hXWzt8GfxR9xy8ezx+Mx2ZfXVcMzs+e6073nq4el0Z/U0fXTW/Ox5ceR0ZXrf7K3V8XVXdNNf6WbwU+929vi5Ts1h6PTlbELydcxZZXJn7GTjRt3++/DvzPLydunzWe1TxWro8flM7W5610/rSOnAem9U03TNdHX0anHp11xoSUo75G3k6ne/b9hx3PT6VVhr0WrseGTY4k5rdWfmbXhWcq3s9Px18O7Y3eRd9l3sNtn3emn1TXr1bvHhcvJp+j1PzOpzcm1jq+5r4O3q/ero9GFP7XJxdPTFa+tf3OHocd/EaW7q59T0aW3UYujgU8Lv8eKx7k1Y+fT3c3S87s1palLHaGZZWvPtcurkcmly4fo94HDe8+h450RtdG3qbEBXZ/hu8djrrXRoQNXITcJ4aPTGavbvdDFRau7z+NxcUehU0uNs629z8+lb9l9f+WHq0Pvp/HHk52+zefxw8/Tj+fHucfnl7WjyUtFqZsJH1WtwQWfnYPVG7mE9ZbNfV3HxUdVn+8BEb/Pn6mL04upl4VFsb/Vn8Nz30m728WlcYHThcmPz22L38/LMX8PG/m7xc53e8XVVxFJnZU+y+EdecNHK1PR4bOvwemr24llvZvHE02367WVbZzByb2TsZnls9+lsyOnwaVv0QdryxSrxaWVB8njm8UL49HFhUvhM4UQ9PVlkb+nZTHTwZHT09G151fL08PDmbM1HcfZqXuBtbOj18uL18eFj+Mz44qG69eJvtulz6/tMXvDnU9JYN21oRLXVYFVxbuC2dWVu6va7fkVk7nhH6GhuSGF28O/qdnJkYtlybOncZNbOa9pGveFteEBh3WhVQfJ08nnTd1FhcEQ9cnR51e1vZffSbWpp+PzOenV2d3JyaMlieGh17+Vi4O5o6WtD63Tpx1Nh0UXwylrI+PL2Tzxv7vDtUupT7fLMylldX1RYblLv4Ozp2mzASPTxcHdhSHBKaHZy9dNp6WFCXfXnbHRuZlC+cTk5vWVoQ/LjXnT+6e31cVbb2fb6Scho+fbwcHjqY+xt72FzV1ftcW5w7XNmmHb4+cz3aPLqcGVx8GTv62MwU/FWxHhg8UNxY+9ySel01Gw6a2BnaqxQ6vlY6W/mdnX4QGZqcm3XePHxbL3nbs3rcuLpbNrdedNI8W7k6PJrvt1sVLnscWBER3R4cURR4G5rRPPZyOgewufz4/bjamb2cnFkbGJrT3breGhw4vU3+HJr9vNy51NyaVrz6elxb0vNYUM8YupkMmv16XNF9/fhadVSOvn4U1nlaPh3XW7g8vBH49Xt8klx8fNyenR1UnJ3den1cHHuW01j+uL0YvK6ZtZ5SHDy5sHk8XBwTtzY+GgyX9j7Xs3Z2ff42nVC88x3den1d/VH7XZxZPbx9VJyfmframby6nlu83JzaOvvyGX30cNv+2FGZ+TraDJdR+I7z3Jd8+VEyfDzd+xq0uZxfklictrvy8jwbXDf+vPFaO1d+Xnr7+3k4XNr7drs+DT06MzGcnLxx+xraMizdetzZ9fvV9LrSdJpemlh43HoaG752fDt/c1t6fzk6XRxLOlpdHNp/W9f1nlm8+36eHHL3ObTufph/Lp243F1sHL47+66501w785O8nfs7uVf8W9y4Wbq49r74XPwY2l1ebBvcfTub2jx9/Fq4fJsbFpzvGhxBTz0YnWpd/F1airx7V/rQu5c+HnUeeL+aPDfZ1/r+tRRW+BiO9tz6/VTZXw56b53c3V47dze4+zd+vLq6Tri49Q+/GThK3FpuGYj72dVexvjOmlw/tdg/WvscHDzZGpR72t97cfM9fNW9/f4yer86Gz2cmNbderb6N3u9GpM9OvWIvhWaUt3b2DqwfbtaWrJWdZp5ls/83fl6HHp+PNybuBvYXhFZm5p2uHUc5lfcmn03Wxb0ez363FY6/hdImpu0cD+Um9NdHPU8EX272xbS21eZvLQyXLq0enbdG3t6drr7/B30HXz9WpkbP7I6OnvVtbx6nFqSnbqq/HwaMzn8E03bPBjxvN153RNxfNbZ67t+OlwQlPy+G9rbXtq4uPjaXpsfUNpZer5cm5puWB73+dsMm/k7Wvq6/Di8+nKTnNPtPpy8chceFJLyWdh4u/S7GHuzDfRdnJWYXNZ7OzhafnsY3esYnL+ckTka7LJduLicHHSanLl5eN36Wt1QN1Jz0F37vb9YnLg785sbOTnUeFW3O9VRvb64vDcX+1ta1bid2rpuPHxevlv4XRHXnj2XvZt5GzyXvDm8uNm+7Hv70Ae++fiTOK9WWrE8elx8MlkU3HlVjbw8W14Ym7w4Hh4+Wnp8yFpZHFK6Wz9LW5f629ybPDtXebjd2ns8ug1c2tIt/VxaMl0YHfsSe9r1/RN5kzpZ0HHbnjx7fppXdFe8urh4eJHZHQpcG/uXcz3+fTxa3la5uPrcux4VfJwxvBqQ0PuZ3A/b2hf8UtpZuNo2ExhcPLAU/rv9Gn1cmP6c+1k2GD6wW75eXbsznm37Ov4aWjm2fBoad3n7vhw8sXtcUQt8PZswnric3SsbGP4crx02WnqIUX4cXnteGxx1fNZ3UFrfDfn73rnYWl8vOlU4+3vd2pZ5G9v6WnSaebAYOzEx0HsdLZxZ3voT9vycWW8cdtxWUy4XnlqW3Xu+mnjbGNda3gG6nL15/JtdLnpYlXxaXNycnZlbOdu8nRdMFpkULpoW87BVuvR/MFxX1Ljx2rX7HHMxfB22XFkbEN49vH2YMh+QmhiyurvcHNI63pLaPZs8ubu5u9qc2ng+bNictS3b/RyQPNqZ+s10Or0QD7o19znS8bfenH1WuNy3XLz82FseUpvaWZxaHb4ynjx6e1JXObeaG5vcHmyTOr6Z3A4RfjZZTXi0eb4Q3rtdXJUdEr2Zr7CbGxxcWt27U9tbGJK7XlG8vRH7WPv4zzi/Tvqw3xpb23lU+3Z5nlSRvTrrUD7PDy6ddDpbq3q8Ov6T/vRcXA8WHZ18HXx2XJn7dNu7ef0QNv0enhz3vH90/XgY2HTae1waWDg8OnuW0/TfFU/dvnqt2t0WlG9WNpv0w=="
A_SCALES_B64 = "eHh7eHh3eHh8eHh4eHh6fHh7eHh7end4eHh4end4eHh4eHd4eH54eH14eHd3d3h4eHh3eHh4eH54eHh4d3h3eHh4fHh4fnh4eHl3eHh4d314eHh4eHd4fHh4eHh4eHx3fHh8eHh4d3d7eHh9eHh4eHh4eHh4eHh4eHh4fXd4fXg="
B_SCALES_B64 = "d3h3e3p7eHh4eHh9eHh4eHh3eHd4eHh4d3h4eHx4eHh4eHh4eH54d3h4eHh4eHh6eHt4fnd4d3p4fHh3eHh4eHx4e3h4fHx4d3h4eHh4eHh4eHh4d3x4eHd4eHh3fXh3eHh4eHh4eHh4eHh4eHh9eXh8fnd4eH14eHh4fXh4eHg="


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__file__ = str(path / "__init__.py")
    module.__path__ = [str(path)]
    return module


def _bootstrap_test_imports() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    modeloptimizer_pkg = _ensure_package("modeloptimizer", MODELOPTIMIZER_DIR)
    kernels_pkg = _ensure_package("modeloptimizer.kernels", KERNELS_DIR)
    mxfp8_pkg = _ensure_package("modeloptimizer.kernels.mxfp8", MXFP8_DIR)
    tests_pkg = _ensure_package("modeloptimizer.kernels.mxfp8.tests", TESTS_DIR)

    modeloptimizer_pkg.kernels = kernels_pkg
    kernels_pkg.mxfp8 = mxfp8_pkg
    mxfp8_pkg.tests = tests_pkg


_bootstrap_test_imports()

from modeloptimizer.kernels.mxfp8.mxfp8_linear import blockwise_mxfp8_gemm
from modeloptimizer.kernels.mxfp8.mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    convert_from_mxfp8,
    is_cdna4,
)


def _decode_uint8_tensor(data_b64: str, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    raw = base64.b64decode(data_b64)
    tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape)
    return tensor.to(device=device)


def _build_inputs(device: torch.device):
    _, n, _ = SHAPE
    a_lp_u8 = _decode_uint8_tensor(A_LP_B64, (SHAPE[0], SHAPE[2]), device)
    b_lp_u8 = _decode_uint8_tensor(B_LP_B64, (SHAPE[2], SHAPE[1]), device)
    a_scales = _decode_uint8_tensor(A_SCALES_B64, (SHAPE[0], SHAPE[2] // BLOCK_SIZE_DEFAULT), device)
    b_scales = _decode_uint8_tensor(B_SCALES_B64, (SHAPE[2] // BLOCK_SIZE_DEFAULT, n), device)
    return a_lp_u8.view(FP8_DTYPE), a_scales, b_lp_u8.view(FP8_DTYPE), b_scales


def _run_kernel(a_lp, a_scales, b_lp, b_scales, use_emulated: bool):
    if use_emulated:
        os.environ["MODELOPTIMIZER_MXFP8_EMULATED_DOT_FORMATS"] = MXFP_FORMAT
    else:
        os.environ.pop("MODELOPTIMIZER_MXFP8_EMULATED_DOT_FORMATS", None)

    return blockwise_mxfp8_gemm(
        a_lp,
        a_scales,
        b_lp,
        b_scales,
        trans_a=False,
        trans_b=False,
        block_size=BLOCK_SIZE_DEFAULT,
        output_dtype=OUTPUT_DTYPE,
        use_accumulator_add=True,
    )


def _build_reference(a_lp, a_scales, b_lp, b_scales):
    a_dq = convert_from_mxfp8(
        a_lp,
        a_scales,
        output_dtype=OUTPUT_DTYPE,
        block_size=BLOCK_SIZE_DEFAULT,
        axis=-1,
        is_2d_block=False,
    )
    b_dq = convert_from_mxfp8(
        b_lp,
        b_scales,
        output_dtype=OUTPUT_DTYPE,
        block_size=BLOCK_SIZE_DEFAULT,
        axis=-2,
        is_2d_block=False,
    )
    return a_dq @ b_dq


def _max_diff(lhs: torch.Tensor, rhs: torch.Tensor):
    diff = (lhs - rhs).abs().float()
    flat_idx = int(diff.argmax().item())
    row, col = divmod(flat_idx, diff.shape[1])
    return diff[row, col].item(), (row, col)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required.")
    if not is_cdna4():
        raise RuntimeError("This repro expects a CDNA4 gfx950 target.")

    device = torch.device("cuda")
    a_lp, a_scales, b_lp, b_scales = _build_inputs(device)
    c_scaled = _run_kernel(a_lp, a_scales, b_lp, b_scales, use_emulated=False)
    c_emulated = _run_kernel(a_lp, a_scales, b_lp, b_scales, use_emulated=True)
    c_reference = _build_reference(a_lp, a_scales, b_lp, b_scales)

    scaled_vs_emulated, idx = _max_diff(c_scaled, c_emulated)
    scaled_vs_reference, _ = _max_diff(c_scaled, c_reference)
    emulated_vs_reference, _ = _max_diff(c_emulated, c_reference)
    row, col = idx

    print(f"shape={SHAPE} format={MXFP_FORMAT} output_dtype={OUTPUT_DTYPE}")
    print("inputs=frozen")
    print(f"max_abs_diff(scaled, emulated)={scaled_vs_emulated}")
    print(f"max_abs_diff(scaled, reference)={scaled_vs_reference}")
    print(f"max_abs_diff(emulated, reference)={emulated_vs_reference}")
    print(f"mismatch_index={(row, col)}")
    print(f"scaled_value={float(c_scaled[row, col])}")
    print(f"emulated_value={float(c_emulated[row, col])}")
    print(f"reference_value={float(c_reference[row, col])}")

    return 1 if scaled_vs_emulated > 0.0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
