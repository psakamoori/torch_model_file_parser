--[[
Purpose:							       
  To parse the torch pre-trained model file (t7)                      
  Prints layer, kernel size, stride and padding information           
How to use							       			
  Replace nn4.samll2.v1.t7 with any of the torch pre-trained   
  model file (t7) 						       
Pre-requisites:						       
  Machine with lua, torch and supporting Luarocks installed           
 Ref: www.torch.ch on how to install                                  
]]--

require 'nn'
require 'dpnn'

model = torch.load('nn4.small2.v1.t7')
for i = 1, #model.modules do
    print(model.modules[i])
end


